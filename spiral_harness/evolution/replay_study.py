"""Complete non-reportable 4 x 2 study over the real automatic search loop.

The fixture is an orchestration proof, not benchmark evidence.  Static cells
close at zero rounds; every non-static cell performs one real proposal,
materialization, and trusted runtime screen before deterministic rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.core.experiment import EXPERIMENT_MANIFEST_MEDIA_TYPE
from spiral_harness.core.models import ArtifactRef, ImmutableModel
from spiral_harness.evolution.models import (
    SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    SearchPolicy,
)
from spiral_harness.evolution.orchestrator import (
    AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE,
    SEARCH_ANALYSIS_PLAN_MEDIA_TYPE,
    AutomaticSearchLoop,
    AutomaticSearchLoopExecution,
    SearchRunAdmissionExpectation,
)
from spiral_harness.evolution.replay_runtime import (
    ReplayRuntime,
    ReplayStrategyPlugin,
)
from spiral_harness.evolution.replay_setup import (
    SEARCH_RUN_SEEDS,
    FrozenReplayFixture,
    build_frozen_replay_fixture,
    freeze_replay_search_runs,
    put_json,
)
from spiral_harness.experiments.baselines import (
    BASELINE_STUDY_PLAN_MEDIA_TYPE,
    LEGACY_BASELINE_KINDS,
    BaselineKind,
)
from spiral_harness.experiments.controller import ExperimentController
from spiral_harness.experiments.search import (
    SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
    SearchController,
    SearchControllerManifest,
    TrustedSearchEventService,
)
from spiral_harness.experiments.study import (
    STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
    STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE,
    StudyController,
    StudyControllerManifest,
    StudyExpectedRunManifest,
    StudyRunKey,
    StudyState,
    TrustedStudyEventService,
)

REPLAY_STUDY_RESULT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.non-reportable-replay-study-result.v1+json"
)


class ReplayStudyRunResult(ImmutableModel):
    """Durable summary for one exact baseline/search-seed cell."""

    key: StudyRunKey
    search_run_ref: ArtifactRef
    search_controller_manifest_ref: ArtifactRef
    experiment_ref: ArtifactRef
    automatic_search_result_ref: ArtifactRef
    final_search_tail_ref: ArtifactRef
    final_experiment_tail_ref: ArtifactRef
    non_reportable_fixture: Literal[True] = True
    completed_rounds: Annotated[int, Field(ge=0, strict=True)]
    proposal_count: Annotated[int, Field(ge=0, strict=True)]
    materialization_call_count: Annotated[int, Field(ge=0, strict=True)]
    screen_count: Annotated[int, Field(ge=0, strict=True)]
    screen_call_count: Annotated[int, Field(ge=0, strict=True)]
    gate_call_count: Literal[0] = 0

    @model_validator(mode="after")
    def _exact_fixture_profile(self) -> Self:
        expected_media_types = (
            (self.search_run_ref, SEARCH_RUN_MANIFEST_MEDIA_TYPE),
            (self.search_controller_manifest_ref, SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE),
            (self.experiment_ref, EXPERIMENT_MANIFEST_MEDIA_TYPE),
            (self.automatic_search_result_ref, AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE),
        )
        if any(ref.media_type != expected for ref, expected in expected_media_types):
            raise ValueError("replay-study run result contains a wrongly typed reference")
        counts = (
            self.completed_rounds,
            self.proposal_count,
            self.materialization_call_count,
            self.screen_count,
            self.screen_call_count,
        )
        expected_counts = (
            (0, 0, 0, 0, 0) if self.key.baseline_kind is BaselineKind.STATIC else (1, 1, 1, 1, 1)
        )
        if counts != expected_counts:
            raise ValueError("replay-study cell did not execute its frozen fixture profile")
        return self


class ReplayStudyResult(ImmutableModel):
    """Persisted proof summary for the complete non-reportable study."""

    schema_version: Literal["1"] = "1"
    non_reportable_fixture: Literal[True] = True
    exact_run_count: Literal[8] = 8
    baseline_study_plan_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    study_controller_manifest_ref: ArtifactRef
    barrier_closed_tail_ref: ArtifactRef
    sealed_authorized_tail_ref: ArtifactRef
    sealed_authorization_ref: ArtifactRef
    runs: tuple[ReplayStudyRunResult, ...]

    @model_validator(mode="after")
    def _complete_four_by_two_schedule(self) -> Self:
        exact_types = (
            (self.baseline_study_plan_ref, BASELINE_STUDY_PLAN_MEDIA_TYPE),
            (self.analysis_plan_ref, SEARCH_ANALYSIS_PLAN_MEDIA_TYPE),
            (self.study_controller_manifest_ref, STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE),
            (self.sealed_authorization_ref, STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE),
        )
        if any(ref.media_type != expected for ref, expected in exact_types):
            raise ValueError("replay-study result contains a wrongly typed control ref")
        if len(self.runs) != self.exact_run_count:
            raise ValueError("replay study must contain exactly eight runs")
        coordinates = tuple((run.key.baseline_kind, run.key.search_run_seed) for run in self.runs)
        expected = {(kind, seed) for kind in LEGACY_BASELINE_KINDS for seed in SEARCH_RUN_SEEDS}
        if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
            raise ValueError("replay study does not cover the exact four-by-two schedule")
        manifests = tuple(run.search_controller_manifest_ref for run in self.runs)
        if len({ref.sha256 for ref in manifests}) != self.exact_run_count:
            raise ValueError("replay study reused a search-controller manifest")
        return self


@dataclass(frozen=True, slots=True)
class ReplayStudyRunExecution:
    """Live capabilities retained for final sealed re-verification."""

    result: ReplayStudyRunResult
    automatic_search_loop: AutomaticSearchLoop
    automatic_search: AutomaticSearchLoopExecution
    search_controller: SearchController
    experiment_controller: ExperimentController
    runtime: ReplayRuntime


@dataclass(frozen=True, slots=True)
class ReplayStudyExecution:
    """In-memory execution and its durable content-addressed summary."""

    fixture: FrozenReplayFixture
    result_ref: ArtifactRef
    result: ReplayStudyResult
    study_controller: StudyController
    runs: tuple[ReplayStudyRunExecution, ...]


def run_non_reportable_replay_study(root: str | Path) -> ReplayStudyExecution:
    """Execute eight admitted loops, close their barrier, and authorize sealed."""

    fixture = build_frozen_replay_fixture(root)
    frozen_runs = freeze_replay_search_runs(fixture)
    study_service = TrustedStudyEventService()
    study_manifest_ref = put_json(
        fixture.store,
        StudyControllerManifest(
            baseline_study_plan_ref=fixture.baseline_study_plan_ref,
            analysis_plan_ref=fixture.analysis_plan_ref,
            expected_run_manifests=tuple(
                StudyExpectedRunManifest(key=key, search_run_ref=search_run_ref)
                for key, _, search_run_ref in frozen_runs
            ),
            study_id="non-reportable-four-by-two-replay-study-v1",
            journal_attestor_id=study_service.attestor_id,
        ),
        media_type=STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
    )
    study_controller = StudyController(
        fixture.store,
        study_controller_manifest_ref=study_manifest_ref,
        event_service=study_service,
    )
    study_tail = study_controller.freeze()
    study_tail = study_controller.start_collection(previous_tail_ref=study_tail)

    executions: list[ReplayStudyRunExecution] = []
    for key, search_run, search_run_ref in frozen_runs:
        policy = fixture.store.get_json(search_run.search_policy_ref, SearchPolicy)
        experiment_controller = ExperimentController(
            fixture.store,
            experiment_ref=fixture.experiment_ref,
            gate_batch_verifier=fixture.gate_batch_service.verification_capability,
            mechanism_evidence_verifier=(
                fixture.mechanism_evidence_service.verification_capability
            ),
        )
        experiment_tail = experiment_controller.freeze_experiment()
        experiment_controller.start_search(previous_tail_ref=experiment_tail)

        search_service = TrustedSearchEventService()
        search_manifest_ref = put_json(
            fixture.store,
            SearchControllerManifest(
                search_run_ref=search_run_ref,
                study_ref=fixture.baseline_study_plan_ref,
                experiment_ref=fixture.experiment_ref,
                run_id=f"{key.baseline_kind.value}-seed-{key.search_run_seed}",
                baseline_kind=key.baseline_kind,
                search_seed=key.search_run_seed,
                initial_champion_harness_ref=fixture.seed_harness_ref,
                analysis_plan_ref=fixture.analysis_plan_ref,
                max_rounds=policy.max_rounds,
                max_gate_nominations=policy.max_gate_queries,
                patience_rounds=policy.patience_rounds,
                max_consecutive_declines=policy.max_consecutive_declines,
                journal_attestor_id=search_service.attestor_id,
            ),
            media_type=SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
        )
        search_controller = SearchController(
            fixture.store,
            controller_manifest_ref=search_manifest_ref,
            event_service=search_service,
            terminal_authorization_verifier=(
                None if key.baseline_kind is BaselineKind.STATIC else experiment_controller
            ),
        )
        runtime = ReplayRuntime(fixture)
        plugin = (
            ReplayStrategyPlugin(
                baseline_kind=key.baseline_kind,
                manifest_ref=fixture.plugin_refs[key.baseline_kind],
                implementation_ref=fixture.implementation_refs[key.baseline_kind],
                failure_signature_ref=fixture.feedback.failure_signature_ref,
                evidence_packet_ref=fixture.feedback.evidence_packet_ref,
                diagnostic_cluster_ref=fixture.feedback.diagnostic_cluster_ref,
            )
            if key.baseline_kind in {BaselineKind.PROMPT_ONLY, BaselineKind.EVIDENCE_TARGETED}
            else None
        )
        automatic_search_loop = AutomaticSearchLoop(
            fixture.store,
            runtime=runtime,
            lifecycle=experiment_controller,
            objective_aggregate_verifier=(fixture.objective_service.verification_capability),
            strategy_feedback_verifier=(fixture.feedback_service.verification_capability),
            plugin=plugin,
            non_reportable_fixture=True,
        )
        automatic_search = automatic_search_loop.run(
            search_run_ref=search_run_ref,
            controller=search_controller,
            expectation=SearchRunAdmissionExpectation(
                baseline_study_plan_ref=fixture.baseline_study_plan_ref,
                experiment_ref=fixture.experiment_ref,
                baseline_kind=key.baseline_kind,
                search_run_seed=key.search_run_seed,
            ),
        )
        automatic_result = automatic_search.result
        run_result = ReplayStudyRunResult(
            key=key,
            search_run_ref=search_run_ref,
            search_controller_manifest_ref=search_manifest_ref,
            experiment_ref=fixture.experiment_ref,
            automatic_search_result_ref=automatic_search.result_ref,
            final_search_tail_ref=automatic_result.final_search_tail_ref,
            final_experiment_tail_ref=automatic_result.final_experiment_tail_ref,
            completed_rounds=automatic_result.completed_rounds,
            proposal_count=automatic_result.proposal_count,
            materialization_call_count=runtime.materialization_call_count,
            screen_count=automatic_result.screen_count,
            screen_call_count=runtime.screen_call_count,
            gate_call_count=runtime.gate_call_count,
        )
        study_tail = study_controller.register_run_closure(
            previous_tail_ref=study_tail,
            automatic_search_loop=automatic_search_loop,
            automatic_search_result_ref=automatic_search.result_ref,
            search_controller=search_controller,
            search_selection_tail_ref=automatic_result.final_search_tail_ref,
            experiment_controller=experiment_controller,
            experiment_selection_tail_ref=automatic_result.final_experiment_tail_ref,
        )
        executions.append(
            ReplayStudyRunExecution(
                result=run_result,
                automatic_search_loop=automatic_search_loop,
                automatic_search=automatic_search,
                search_controller=search_controller,
                experiment_controller=experiment_controller,
                runtime=runtime,
            )
        )

    barrier_tail = study_controller.close_selection_barrier(previous_tail_ref=study_tail)
    sealed_tail = study_controller.authorize_sealed(previous_tail_ref=barrier_tail)
    authorization = study_controller.verify_sealed_authorization(sealed_tail)
    snapshot = study_controller.snapshot
    if (
        study_controller.state is not StudyState.SEALED_AUTHORIZED
        or snapshot is None
        or snapshot.sealed_authorization_ref is None
        or authorization.bindings != snapshot.bindings
    ):
        raise RuntimeError("replay study sealed authorization did not reverify")

    result = ReplayStudyResult(
        baseline_study_plan_ref=fixture.baseline_study_plan_ref,
        analysis_plan_ref=fixture.analysis_plan_ref,
        study_controller_manifest_ref=study_manifest_ref,
        barrier_closed_tail_ref=barrier_tail,
        sealed_authorized_tail_ref=sealed_tail,
        sealed_authorization_ref=snapshot.sealed_authorization_ref,
        runs=tuple(execution.result for execution in executions),
    )
    result_ref = put_json(
        fixture.store,
        result,
        media_type=REPLAY_STUDY_RESULT_MEDIA_TYPE,
    )
    return ReplayStudyExecution(
        fixture=fixture,
        result_ref=result_ref,
        result=result,
        study_controller=study_controller,
        runs=tuple(executions),
    )


__all__ = [
    "REPLAY_STUDY_RESULT_MEDIA_TYPE",
    "ReplayStudyExecution",
    "ReplayStudyResult",
    "ReplayStudyRunExecution",
    "ReplayStudyRunResult",
    "run_non_reportable_replay_study",
]
