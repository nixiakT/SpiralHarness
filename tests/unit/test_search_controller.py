from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from spiral_harness.core.experiment import EXPERIMENT_MANIFEST_MEDIA_TYPE
from spiral_harness.core.lifecycle import CandidateState
from spiral_harness.core.models import ArtifactRef, ComponentKind
from spiral_harness.evolution.models import (
    SEARCH_POLICY_MEDIA_TYPE,
    SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    SEARCH_STOPPING_POLICY_MEDIA_TYPE,
    STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
    SearchPolicy,
    SearchRunManifest,
    SearchStoppingPolicy,
    expected_strategy_feedback,
    expected_strategy_mutation,
)
from spiral_harness.evolution.seeds import derive_strategy_seed
from spiral_harness.experiments.baselines import (
    BASELINE_STUDY_PLAN_MEDIA_TYPE,
    BaselineKind,
    FrozenMutationPolicy,
)
from spiral_harness.experiments.controller import (
    EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE,
    TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE,
    TerminalTransitionAuthorization,
)
from spiral_harness.experiments.decision import (
    GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    TERMINAL_DECISION_REPORT_MEDIA_TYPE,
)
from spiral_harness.experiments.search import (
    AGGREGATE_FEEDBACK_DISCLOSURE_MEDIA_TYPE,
    SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
    SEARCH_JOURNAL_EVENT_MEDIA_TYPE,
    AggregateFeedbackDisclosure,
    RoundStage,
    SearchController,
    SearchControllerError,
    SearchControllerManifest,
    SearchEventAttestationError,
    SearchEventKind,
    SearchJournal,
    SearchJournalCycleError,
    SearchJournalEvent,
    SearchJournalEventContent,
    SearchJournalIntegrityError,
    SearchState,
    SearchStopError,
    SearchStopReason,
    StaleSearchTailError,
    TrustedSearchEventService,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.storage.journal import JOURNAL_ENTRY_MEDIA_TYPE
from spiral_harness.verification.models import (
    ComparisonMetrics,
    ComparisonResult,
    ConfidenceInterval,
    Decision,
    GateCheck,
    GateCheckOutcome,
    GateDecision,
    PairingAudit,
)


def artifact(digit: str, *, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=1, media_type=media_type)


def ref_key(ref: ArtifactRef) -> tuple[str, int, str]:
    return ref.sha256, ref.size, ref.media_type


@dataclass
class AllowlistedTerminalVerifier:
    allowed: dict[tuple[str, int, str], TerminalTransitionAuthorization] = field(
        default_factory=dict
    )

    def authorize(self, ref: ArtifactRef, authorization: TerminalTransitionAuthorization) -> None:
        self.allowed[ref_key(ref)] = authorization

    def verify_terminal_authorization(self, ref: ArtifactRef) -> TerminalTransitionAuthorization:
        try:
            return self.allowed[ref_key(ref)]
        except KeyError as exc:
            raise ValueError("terminal authorization ref was not issued by controller") from exc


def put(store: ArtifactStore, label: str) -> ArtifactRef:
    return store.put_json({"label": label})


def build_controller(
    tmp_path: Path,
    *,
    baseline_kind: BaselineKind = BaselineKind.EVIDENCE_TARGETED,
    max_rounds: int = 3,
    max_gate_nominations: int = 3,
    patience_rounds: int = 3,
    max_consecutive_declines: int = 3,
) -> SimpleNamespace:
    store = ArtifactStore(tmp_path)
    service = TrustedSearchEventService(secret=b"search-controller-test-secret!!" + b"x" * 8)
    verifier = AllowlistedTerminalVerifier()
    study_ref = store.put_json({"label": "study"}, media_type=BASELINE_STUDY_PLAN_MEDIA_TYPE)
    experiment_ref = store.put_json(
        {"label": "experiment"}, media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE
    )
    champion_ref = put(store, "seed-harness")
    analysis_plan_ref = put(store, "pre-registered-analysis")
    if baseline_kind is BaselineKind.STATIC:
        max_rounds = 0
        max_gate_nominations = 0
        patience_rounds = 0
        max_consecutive_declines = 0
    max_gate_nominations = min(max_gate_nominations, max_rounds)
    is_static = baseline_kind is BaselineKind.STATIC
    max_proposals = 0 if is_static else max(max_gate_nominations, max_rounds * 2)
    max_diagnoses = max_rounds * 2 if baseline_kind is BaselineKind.EVIDENCE_TARGETED else 0
    search_policy = SearchPolicy(
        baseline_kind=baseline_kind,
        mutation_policy=FrozenMutationPolicy(
            grammar_version="prompt-only-test-v1",
            allowed_component_kinds=(ComponentKind.PROMPT,),
            max_artifact_size_bytes=4096,
        ),
        available_feedback=tuple(expected_strategy_feedback(baseline_kind)),
        mutation=expected_strategy_mutation(baseline_kind),
        max_rounds=max_rounds,
        max_gate_queries=max_gate_nominations,
        patience_rounds=patience_rounds,
        max_consecutive_declines=max_consecutive_declines,
        max_diagnoses=max_diagnoses,
        max_proposals=max_proposals,
        max_screens=max_proposals,
        max_proposals_per_round=0 if is_static else 1,
        max_candidates_screened_per_round=0 if is_static else 1,
        family_alpha=0.05,
        gate_confidence_level=(None if is_static else 1.0 - (0.05 / max_gate_nominations)),
    )
    search_policy_ref = store.put_json(search_policy, media_type=SEARCH_POLICY_MEDIA_TYPE)
    strategy_plugin_ref = store.put_json(
        {"label": "strategy-plugin"}, media_type=STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE
    )
    stopping_policy = SearchStoppingPolicy(
        baseline_kind=baseline_kind,
        max_rounds=max_rounds,
        max_gate_queries=max_gate_nominations,
        patience_rounds=patience_rounds,
        max_consecutive_declines=max_consecutive_declines,
    )
    stopping_policy_ref = store.put_json(
        stopping_policy, media_type=SEARCH_STOPPING_POLICY_MEDIA_TYPE
    )
    proposal_master_seed = 991
    search_seed = 701
    search_run = SearchRunManifest(
        baseline_study_plan_ref=study_ref,
        experiment_ref=experiment_ref,
        search_policy_ref=search_policy_ref,
        search_policy_fingerprint=search_policy_ref.sha256,
        strategy_plugin_ref=strategy_plugin_ref,
        strategy_plugin_fingerprint=strategy_plugin_ref.sha256,
        analysis_plan_ref=analysis_plan_ref,
        stopping_policy_ref=stopping_policy_ref,
        stopping_policy_fingerprint=stopping_policy_ref.sha256,
        seed_harness_ref=champion_ref,
        baseline_kind=baseline_kind,
        baseline_plan_fingerprint=study_ref.sha256,
        proposal_master_seed=proposal_master_seed,
        search_run_seed=search_seed,
        repeat_seeds=(11, 12),
        strategy_seed=derive_strategy_seed(
            proposal_master_seed=proposal_master_seed,
            search_run_seed=search_seed,
            baseline_kind=baseline_kind,
        ),
    )
    search_run_ref = store.put_json(search_run, media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE)
    manifest = SearchControllerManifest(
        search_run_ref=search_run_ref,
        study_ref=study_ref,
        experiment_ref=experiment_ref,
        run_id="run-07",
        baseline_kind=baseline_kind,
        search_seed=search_seed,
        initial_champion_harness_ref=champion_ref,
        analysis_plan_ref=analysis_plan_ref,
        max_rounds=max_rounds,
        max_gate_nominations=max_gate_nominations,
        patience_rounds=patience_rounds,
        max_consecutive_declines=max_consecutive_declines,
        journal_attestor_id=service.attestor_id,
    )
    manifest_ref = store.put_json(manifest, media_type=SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE)
    controller = SearchController(
        store,
        controller_manifest_ref=manifest_ref,
        event_service=service,
        terminal_authorization_verifier=(
            None if baseline_kind is BaselineKind.STATIC else verifier
        ),
    )
    return SimpleNamespace(
        store=store,
        service=service,
        verifier=verifier,
        manifest=manifest,
        search_run=search_run,
        search_policy=search_policy,
        stopping_policy=stopping_policy,
        manifest_ref=manifest_ref,
        controller=controller,
        champion_ref=champion_ref,
    )


def start_round(env: SimpleNamespace) -> tuple[ArtifactRef, ArtifactRef]:
    controller = env.controller
    tail = controller.freeze()
    tail = controller.start_search(previous_tail_ref=tail)
    tail = controller.open_round(previous_tail_ref=tail)
    evidence_ref = put(env.store, "exploration-evidence")
    tail = controller.record_evidence(previous_tail_ref=tail, evidence_packet_ref=evidence_ref)
    candidate_ref = put(env.store, "candidate-0")
    pool_ref = put(env.store, "candidate-pool")
    tail = controller.record_candidate_pool(
        previous_tail_ref=tail,
        candidate_pool_ref=pool_ref,
        candidate_refs=(candidate_ref,),
    )
    screen_ref = put(env.store, "screen")
    tail = controller.record_screen(
        previous_tail_ref=tail,
        screen_report_ref=screen_ref,
        eligible_candidate_refs=(candidate_ref,),
    )
    return tail, candidate_ref


def gate_decision(
    *, decision: Decision, parent_ref: ArtifactRef, child_ref: ArtifactRef
) -> GateDecision:
    metrics = ComparisonMetrics(
        n_valid_pairs=4,
        n_tasks=2,
        parent_mean=0.5,
        candidate_mean=0.75,
        mean_delta=0.25,
        confidence_interval=ConfidenceInterval(confidence_level=0.95, lower=0.10, upper=0.40),
        wins=2,
        ties=0,
        losses=0,
        regression_rate=0.0,
        worst_task_delta=0.10,
        parent_tokens_mean=10.0,
        candidate_tokens_mean=12.0,
        tokens_ratio=1.2,
        parent_latency_ms_mean=20.0,
        candidate_latency_ms_mean=18.0,
        latency_ratio=0.9,
        parent_tool_calls_mean=1.0,
        candidate_tool_calls_mean=1.0,
        tool_calls_ratio=1.0,
        task_comparisons=(),
        slices={},
    )
    outcome = {
        Decision.PROMOTE: GateCheckOutcome.PASS,
        Decision.REJECT: GateCheckOutcome.FAIL,
        Decision.INCONCLUSIVE: GateCheckOutcome.INCONCLUSIVE,
    }[decision]
    return GateDecision(
        decision=decision,
        gate_version="test-v1",
        gate_config_sha256="c" * 64,
        parent_harness_id=parent_ref.sha256,
        candidate_harness_id=child_ref.sha256,
        checks=tuple(
            GateCheck(
                name=name,
                outcome=outcome,
                reasons=("raw reason must be redacted",),
                metrics={"task-id": "must-not-leak"},
            )
            for name in (
                "integrity",
                "sample_size",
                "policy",
                "mechanism",
                "primary_effect",
                "protected_slices",
                "task_regressions",
                "resources",
            )
        ),
        reasons=("decision reason must be redacted",),
        comparison=ComparisonResult(audit=PairingAudit(), metrics=metrics),
    )


def authorize_terminal(
    env: SimpleNamespace,
    *,
    candidate_ref: ArtifactRef,
    decision: Decision,
    parent_ref: ArtifactRef | None = None,
) -> tuple[ArtifactRef, TerminalTransitionAuthorization, ArtifactRef]:
    parent = parent_ref or env.controller.snapshot.champion_harness_ref
    child_ref = put(env.store, f"child-{decision.value}")
    decision_ref = env.store.put_json(
        gate_decision(decision=decision, parent_ref=parent, child_ref=child_ref)
    )
    state = {
        Decision.PROMOTE: CandidateState.PROMOTED,
        Decision.REJECT: CandidateState.REJECTED,
        Decision.INCONCLUSIVE: CandidateState.INCONCLUSIVE,
    }[decision]
    authorization = TerminalTransitionAuthorization(
        experiment_ref=env.manifest.experiment_ref,
        protocol_ref=put(env.store, "protocol"),
        candidate_ref=candidate_ref,
        prior_champion_harness_ref=parent,
        parent_harness_ref=parent,
        child_harness_ref=child_ref,
        evidence_complete_tail_ref=artifact("1", media_type=JOURNAL_ENTRY_MEDIA_TYPE),
        usage_entry_ref=artifact("2", media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE),
        evaluation_ref=artifact("3", media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE),
        terminal_decision_report_ref=artifact("4", media_type=TERMINAL_DECISION_REPORT_MEDIA_TYPE),
        decision_ref=decision_ref,
        gate_terminal_state=state,
        terminal_state=state,
    )
    authorization_ref = env.store.put_json(
        authorization, media_type=TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE
    )
    env.verifier.authorize(authorization_ref, authorization)
    return authorization_ref, authorization, child_ref


def complete_empty_round(env: SimpleNamespace, *, declined: bool) -> ArtifactRef:
    controller = env.controller
    tail = controller.freeze()
    tail = controller.start_search(previous_tail_ref=tail)
    tail = controller.open_round(previous_tail_ref=tail)
    tail = controller.record_evidence(
        previous_tail_ref=tail,
        evidence_packet_ref=put(env.store, "evidence"),
    )
    tail = controller.record_candidate_pool(
        previous_tail_ref=tail,
        candidate_pool_ref=put(env.store, "pool"),
        candidate_refs=(),
        declined=declined,
    )
    tail = controller.record_screen(
        previous_tail_ref=tail,
        screen_report_ref=put(env.store, "screen-empty"),
    )
    tail = controller.record_no_candidate(previous_tail_ref=tail, reason="no valid proposal")
    return controller.complete_no_candidate_round(previous_tail_ref=tail)


def test_static_run_closes_at_zero_rounds_with_prefrozen_analysis(tmp_path: Path) -> None:
    env = build_controller(tmp_path, baseline_kind=BaselineKind.STATIC)
    tail = env.controller.freeze()
    tail = env.controller.start_search(previous_tail_ref=tail)
    closed_tail = env.controller.close_selection(previous_tail_ref=tail)

    snapshot = env.controller.verify_selection_closure(closed_tail)
    closure = env.store.get_json(snapshot.selection_closure_ref)
    assert snapshot.state is SearchState.SELECTION_CLOSED
    assert snapshot.completed_rounds == 0
    assert snapshot.stop_reasons == (SearchStopReason.STATIC,)
    assert closure["analysis_plan_ref"]["sha256"] == env.manifest.analysis_plan_ref.sha256
    assert snapshot.proposal_api_open is False
    with pytest.raises(SearchControllerError, match="SEARCHING"):
        env.controller.open_round(previous_tail_ref=closed_tail)


def test_round_stage_machine_nominates_once_and_promotes_only_after_verified_terminal(
    tmp_path: Path,
) -> None:
    env = build_controller(tmp_path, max_rounds=1, patience_rounds=1, max_consecutive_declines=1)
    screen_tail, candidate_ref = start_round(env)
    nominated_tail = env.controller.nominate(
        previous_tail_ref=screen_tail,
        nomination_ref=put(env.store, "nomination"),
        candidate_ref=candidate_ref,
    )
    assert env.controller.snapshot.round_stage is RoundStage.NOMINATED
    assert env.controller.snapshot.champion_harness_ref == env.champion_ref
    with pytest.raises(SearchControllerError, match="SCREENED"):
        env.controller.nominate(
            previous_tail_ref=nominated_tail,
            nomination_ref=put(env.store, "second-nomination"),
            candidate_ref=candidate_ref,
        )

    authorization_ref, _, child_ref = authorize_terminal(
        env, candidate_ref=candidate_ref, decision=Decision.PROMOTE
    )
    completed_tail = env.controller.complete_nominated_round(
        previous_tail_ref=nominated_tail,
        terminal_authorization_ref=authorization_ref,
    )
    snapshot = env.controller.snapshot
    assert snapshot.champion_harness_ref == child_ref
    assert snapshot.champion_candidate_ref == candidate_ref
    assert snapshot.completed_rounds == 1
    assert snapshot.gate_nominations == 1
    assert snapshot.round_stage is None

    closed_tail = env.controller.close_selection(previous_tail_ref=completed_tail)
    assert (
        env.controller.verify_selection_closure(closed_tail).state is SearchState.SELECTION_CLOSED
    )
    kinds = tuple(
        env.store.get_json(ref, SearchJournalEvent).kind
        for ref in env.controller.snapshot.event_refs
    )
    assert kinds == (
        SearchEventKind.FROZEN,
        SearchEventKind.SEARCH_STARTED,
        SearchEventKind.ROUND_OPENED,
        SearchEventKind.EVIDENCE_RECORDED,
        SearchEventKind.CANDIDATE_POOL_RECORDED,
        SearchEventKind.SCREEN_RECORDED,
        SearchEventKind.CANDIDATE_NOMINATED,
        SearchEventKind.NOMINATED_ROUND_COMPLETED,
        SearchEventKind.SELECTION_CLOSED,
    )


def test_feedback_is_derived_redacted_and_not_accepted_from_caller(tmp_path: Path) -> None:
    env = build_controller(tmp_path, max_rounds=1)
    screen_tail, candidate_ref = start_round(env)
    nominated_tail = env.controller.nominate(
        previous_tail_ref=screen_tail,
        nomination_ref=put(env.store, "nomination"),
        candidate_ref=candidate_ref,
    )
    authorization_ref, _, _ = authorize_terminal(
        env, candidate_ref=candidate_ref, decision=Decision.REJECT
    )
    with pytest.raises(TypeError, match="aggregate_feedback"):
        env.controller.complete_nominated_round(
            previous_tail_ref=nominated_tail,
            terminal_authorization_ref=authorization_ref,
            aggregate_feedback={"task-id": "leak"},  # type: ignore[call-arg]
        )

    env.controller.complete_nominated_round(
        previous_tail_ref=nominated_tail,
        terminal_authorization_ref=authorization_ref,
    )
    disclosure_ref = env.controller.snapshot.last_feedback_disclosure_ref
    assert disclosure_ref is not None
    disclosure = env.store.get_json(disclosure_ref, AggregateFeedbackDisclosure)
    assert disclosure_ref.media_type == AGGREGATE_FEEDBACK_DISCLOSURE_MEDIA_TYPE
    assert disclosure.summary.decision is Decision.REJECT
    assert disclosure.summary.n_valid_pairs == 4
    assert disclosure.summary.mean_score_delta == 0.25
    assert disclosure.summary.confidence_lower_bound == 0.10
    assert disclosure.summary.regression_rate == 0.0
    assert disclosure.gate_version == disclosure.summary.gate_version == "test-v1"
    assert disclosure.gate_config_sha256 == disclosure.summary.gate_config_sha256 == "c" * 64
    serialized = env.store.get_json(disclosure_ref)
    assert "task_comparisons" not in str(serialized)
    assert "slices" not in serialized["summary"]
    assert "raw reason" not in str(serialized)
    assert "task-id" not in str(serialized)


def test_superseded_local_promotion_discloses_authorized_inconclusive_outcome(
    tmp_path: Path,
) -> None:
    env = build_controller(tmp_path, max_rounds=1)
    screen_tail, candidate_ref = start_round(env)
    nominated_tail = env.controller.nominate(
        previous_tail_ref=screen_tail,
        nomination_ref=put(env.store, "nomination"),
        candidate_ref=candidate_ref,
    )
    _, authorization, _ = authorize_terminal(
        env,
        candidate_ref=candidate_ref,
        decision=Decision.PROMOTE,
    )
    superseded_ref = env.store.put_json(
        {"reason": "champion already advanced"},
        media_type=SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE,
    )
    authorization = authorization.model_copy(
        update={
            "terminal_state": CandidateState.INCONCLUSIVE,
            "superseded_report_ref": superseded_ref,
        }
    )
    authorization_ref = env.store.put_json(
        authorization,
        media_type=TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE,
    )
    env.verifier.authorize(authorization_ref, authorization)

    env.controller.complete_nominated_round(
        previous_tail_ref=nominated_tail,
        terminal_authorization_ref=authorization_ref,
    )
    disclosure_ref = env.controller.snapshot.last_feedback_disclosure_ref
    assert disclosure_ref is not None
    disclosure = env.store.get_json(disclosure_ref, AggregateFeedbackDisclosure)
    assert disclosure.terminal_state is CandidateState.INCONCLUSIVE
    assert disclosure.summary.decision is Decision.INCONCLUSIVE
    assert disclosure.summary.gate_version == "test-v1"
    assert disclosure.summary.gate_config_sha256 == "c" * 64
    assert disclosure.champion_harness_ref == env.champion_ref


def test_unissued_terminal_authorization_cannot_advance_champion(tmp_path: Path) -> None:
    env = build_controller(tmp_path)
    screen_tail, candidate_ref = start_round(env)
    nominated_tail = env.controller.nominate(
        previous_tail_ref=screen_tail,
        nomination_ref=put(env.store, "nomination"),
        candidate_ref=candidate_ref,
    )
    authorization_ref, authorization, _ = authorize_terminal(
        env, candidate_ref=candidate_ref, decision=Decision.PROMOTE
    )
    env.verifier.allowed.clear()

    with pytest.raises(SearchJournalIntegrityError, match="was not issued"):
        env.controller.complete_nominated_round(
            previous_tail_ref=nominated_tail,
            terminal_authorization_ref=authorization_ref,
        )
    assert env.controller.snapshot.champion_harness_ref == env.champion_ref

    forged = authorization.model_copy(update={"protocol_ref": put(env.store, "forged-protocol")})
    forged_ref = env.store.put_json(forged, media_type=TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE)
    with pytest.raises(SearchJournalIntegrityError, match="was not issued"):
        env.controller.complete_nominated_round(
            previous_tail_ref=nominated_tail,
            terminal_authorization_ref=forged_ref,
        )


def test_non_static_controller_requires_terminal_verification_capability(
    tmp_path: Path,
) -> None:
    env = build_controller(tmp_path)
    with pytest.raises(SearchControllerError, match="terminal authorization verifier"):
        SearchController(
            env.store,
            controller_manifest_ref=env.manifest_ref,
            event_service=env.service,
        )


def test_controller_manifest_rejoins_the_typed_search_run(tmp_path: Path) -> None:
    env = build_controller(tmp_path)
    drifted = env.manifest.model_copy(
        update={"analysis_plan_ref": put(env.store, "post-hoc-analysis")}
    )
    drifted_ref = env.store.put_json(drifted, media_type=SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE)
    with pytest.raises(SearchJournalIntegrityError, match="search_run_ref"):
        SearchController(
            env.store,
            controller_manifest_ref=drifted_ref,
            event_service=env.service,
            terminal_authorization_verifier=env.verifier,
        )


@pytest.mark.parametrize(
    ("limits", "declined", "decision", "expected"),
    [
        (
            {"max_rounds": 1, "patience_rounds": 1, "max_consecutive_declines": 1},
            False,
            None,
            SearchStopReason.MAX_ROUNDS,
        ),
        (
            {"max_gate_nominations": 1, "patience_rounds": 5},
            False,
            Decision.REJECT,
            SearchStopReason.MAX_GATE_NOMINATIONS,
        ),
        (
            {"patience_rounds": 1},
            False,
            Decision.REJECT,
            SearchStopReason.PATIENCE,
        ),
        (
            {"max_consecutive_declines": 1},
            True,
            None,
            SearchStopReason.MAX_CONSECUTIVE_DECLINES,
        ),
    ],
)
def test_all_frozen_stop_criteria_close_selection(
    tmp_path: Path,
    limits: dict[str, int],
    declined: bool,
    decision: Decision | None,
    expected: SearchStopReason,
) -> None:
    env = build_controller(tmp_path, **limits)
    if decision is None:
        completed_tail = complete_empty_round(env, declined=declined)
    else:
        screen_tail, candidate_ref = start_round(env)
        nominated_tail = env.controller.nominate(
            previous_tail_ref=screen_tail,
            nomination_ref=put(env.store, "nomination"),
            candidate_ref=candidate_ref,
        )
        authorization_ref, _, _ = authorize_terminal(
            env, candidate_ref=candidate_ref, decision=decision
        )
        completed_tail = env.controller.complete_nominated_round(
            previous_tail_ref=nominated_tail,
            terminal_authorization_ref=authorization_ref,
        )
    assert expected in env.controller.snapshot.stop_reasons
    with pytest.raises(SearchStopError, match="stop criterion"):
        env.controller.open_round(previous_tail_ref=completed_tail)
    closed = env.controller.close_selection(previous_tail_ref=completed_tail)
    assert expected in env.controller.verify_selection_closure(closed).stop_reasons


def test_stage_subset_and_stale_tail_violations_fail_closed(tmp_path: Path) -> None:
    env = build_controller(tmp_path)
    root = env.controller.freeze()
    started = env.controller.start_search(previous_tail_ref=root)
    with pytest.raises(StaleSearchTailError, match="stale"):
        env.controller.open_round(previous_tail_ref=root)
    opened = env.controller.open_round(previous_tail_ref=started)
    with pytest.raises(SearchControllerError, match="EVIDENCE_READY"):
        env.controller.record_candidate_pool(
            previous_tail_ref=opened,
            candidate_pool_ref=put(env.store, "premature-pool"),
        )
    evidence = env.controller.record_evidence(
        previous_tail_ref=opened, evidence_packet_ref=put(env.store, "evidence")
    )
    candidate = put(env.store, "candidate")
    pooled = env.controller.record_candidate_pool(
        previous_tail_ref=evidence,
        candidate_pool_ref=put(env.store, "pool"),
        candidate_refs=(candidate,),
    )
    outsider = put(env.store, "outsider")
    with pytest.raises(SearchControllerError, match="subset"):
        env.controller.record_screen(
            previous_tail_ref=pooled,
            screen_report_ref=put(env.store, "bad-screen"),
            eligible_candidate_refs=(outsider,),
        )


def test_hmac_tamper_gap_and_foreign_run_are_detected(tmp_path: Path) -> None:
    env = build_controller(tmp_path)
    root = env.controller.freeze()
    root_event = env.store.get_json(root, SearchJournalEvent)

    tampered = root_event.model_copy(update={"reason": "tampered after signing"})
    tampered_ref = env.store.put_json(tampered, media_type=SEARCH_JOURNAL_EVENT_MEDIA_TYPE)
    with pytest.raises(SearchEventAttestationError, match="HMAC"):
        env.controller.journal.replay(tampered_ref)

    def signed_started(**updates: object) -> ArtifactRef:
        content_values = root_event.content.model_dump(
            mode="python", round_trip=True, warnings="none"
        )
        content_values.update(
            kind=SearchEventKind.SEARCH_STARTED,
            sequence=updates.pop("sequence", 1),
            previous_tail_ref=root,
            from_state=SearchState.FROZEN,
            to_state=SearchState.SEARCHING,
            reason="signed test event",
        )
        content_values.update(updates)
        content = SearchJournalEventContent.model_validate(content_values)
        event = env.service._issue(content)
        return env.store.put_json(event, media_type=SEARCH_JOURNAL_EVENT_MEDIA_TYPE)

    gap_ref = signed_started(sequence=2)
    with pytest.raises(SearchJournalIntegrityError, match="expected 1, got 2"):
        env.controller.journal.replay(gap_ref)
    foreign_ref = signed_started(run_id="foreign-run")
    with pytest.raises(SearchJournalIntegrityError, match="foreign search-run"):
        env.controller.journal.replay(foreign_ref)


class CycleRepository:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def get_json(self, ref: ArtifactRef, model_type: type | None = None) -> object:
        value = self.values[ref.sha256]
        return value if model_type is None else model_type.model_validate(value)

    def get_bytes(self, ref: ArtifactRef) -> bytes:  # pragma: no cover - protocol stub
        raise NotImplementedError

    def put_bytes(
        self, data: bytes, *, media_type: str = "application/octet-stream"
    ) -> ArtifactRef:
        raise NotImplementedError

    def put_json(self, value: object, *, media_type: str = "application/json") -> ArtifactRef:
        raise NotImplementedError


def test_replay_detects_a_cycle_before_following_it_forever(tmp_path: Path) -> None:
    env = build_controller(tmp_path)
    first_ref = artifact("a", media_type=SEARCH_JOURNAL_EVENT_MEDIA_TYPE)
    second_ref = artifact("b", media_type=SEARCH_JOURNAL_EVENT_MEDIA_TYPE)
    common = dict(
        controller_manifest_ref=env.manifest_ref,
        search_run_ref=env.manifest.search_run_ref,
        study_ref=env.manifest.study_ref,
        experiment_ref=env.manifest.experiment_ref,
        run_id=env.manifest.run_id,
        baseline_kind=env.manifest.baseline_kind,
        search_seed=env.manifest.search_seed,
        kind=SearchEventKind.SEARCH_STARTED,
        from_state=SearchState.FROZEN,
        to_state=SearchState.SEARCHING,
        champion_before_ref=env.champion_ref,
        champion_after_ref=env.champion_ref,
        reason="cycle fixture",
    )
    first = env.service._issue(
        SearchJournalEventContent(sequence=2, previous_tail_ref=second_ref, **common)
    )
    second = env.service._issue(
        SearchJournalEventContent(sequence=1, previous_tail_ref=first_ref, **common)
    )
    repository = CycleRepository(
        {
            env.manifest_ref.sha256: env.manifest,
            env.manifest.search_run_ref.sha256: env.search_run,
            env.search_run.search_policy_ref.sha256: env.search_policy,
            env.search_run.stopping_policy_ref.sha256: env.stopping_policy,
            env.manifest.study_ref.sha256: env.store.get_json(env.manifest.study_ref),
            env.manifest.experiment_ref.sha256: env.store.get_json(env.manifest.experiment_ref),
            env.manifest.initial_champion_harness_ref.sha256: env.store.get_json(
                env.manifest.initial_champion_harness_ref
            ),
            env.manifest.analysis_plan_ref.sha256: env.store.get_json(
                env.manifest.analysis_plan_ref
            ),
            first_ref.sha256: first,
            second_ref.sha256: second,
        }
    )
    journal = SearchJournal(
        repository,  # type: ignore[arg-type]
        controller_manifest_ref=env.manifest_ref,
        event_verifier=env.service.verification_capability,
        terminal_authorization_verifier=env.verifier,
    )
    with pytest.raises(SearchJournalCycleError, match="cycle"):
        journal.replay(first_ref)


def test_active_round_can_be_invalidated_and_proposal_api_never_reopens(
    tmp_path: Path,
) -> None:
    env = build_controller(tmp_path)
    root = env.controller.freeze()
    started = env.controller.start_search(previous_tail_ref=root)
    opened = env.controller.open_round(previous_tail_ref=started)
    invalidated = env.controller.invalidate(
        previous_tail_ref=opened,
        invalidation_ref=put(env.store, "integrity-violation"),
        reason="trusted input integrity failure",
    )
    assert env.controller.snapshot.state is SearchState.INVALIDATED
    assert env.controller.snapshot.proposal_api_open is False
    with pytest.raises(SearchControllerError, match="SEARCHING"):
        env.controller.record_evidence(
            previous_tail_ref=invalidated,
            evidence_packet_ref=put(env.store, "late-evidence"),
        )


def test_selection_verifier_rejects_stale_or_non_closed_tails(tmp_path: Path) -> None:
    env = build_controller(tmp_path, baseline_kind=BaselineKind.STATIC)
    frozen = env.controller.freeze()
    searching = env.controller.start_search(previous_tail_ref=frozen)
    with pytest.raises(SearchControllerError, match="not in SELECTION_CLOSED"):
        env.controller.verify_selection_closure(searching)
    closed = env.controller.close_selection(previous_tail_ref=searching)
    assert env.controller.verify_selection_closure(closed).state is SearchState.SELECTION_CLOSED
    invalidated = env.controller.invalidate(
        previous_tail_ref=closed,
        invalidation_ref=put(env.store, "post-close-integrity-failure"),
        reason="selection artifact was corrupted",
    )
    with pytest.raises(StaleSearchTailError, match="stale"):
        env.controller.verify_selection_closure(closed)
    with pytest.raises(SearchControllerError, match="not in SELECTION_CLOSED"):
        env.controller.verify_selection_closure(invalidated)
