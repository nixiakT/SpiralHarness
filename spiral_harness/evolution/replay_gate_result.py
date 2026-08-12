"""Non-reportable replay result that consumes accepted baseline GATE evidence."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.evolution.models import SEARCH_RUN_MANIFEST_MEDIA_TYPE
from spiral_harness.evolution.orchestrator import AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE
from spiral_harness.evolution.replay_setup import SEARCH_RUN_SEEDS
from spiral_harness.evolution.replay_study import (
    REPLAY_STUDY_RESULT_MEDIA_TYPE,
    ReplayStudyResult,
    ReplayStudyRunResult,
)
from spiral_harness.experiments.baseline_gate_acceptance import (
    BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE,
    BaselineGateAcceptanceError,
    BaselineGateStudyAcceptance,
    verify_baseline_gate_study_acceptance,
)
from spiral_harness.experiments.baseline_gate_closure import (
    BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE,
)
from spiral_harness.experiments.baselines import BASELINE_STUDY_PLAN_MEDIA_TYPE, BaselineKind
from spiral_harness.experiments.study import (
    STUDY_BARRIER_CLOSURE_MEDIA_TYPE,
    STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
    STUDY_JOURNAL_EVENT_MEDIA_TYPE,
    STUDY_RUN_CLOSURE_MEDIA_TYPE,
    STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE,
    StudyEventVerificationCapability,
    StudyRunClosure,
    StudyRunKey,
    StudySealedAuthorization,
)
from spiral_harness.storage.protocol import ArtifactRepository

REPLAY_STUDY_GATE_RESULT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.non-reportable-replay-study-gate-result.v1+json"
)


class ReplayStudyGateResultError(RuntimeError):
    """Raised when a replay study result cannot consume accepted GATE evidence."""


class ReplayStudyGateRun(ImmutableModel):
    """One study cell joined from replay summary, study closure, and authorization."""

    key: StudyRunKey
    run_closure_ref: ArtifactRef
    search_run_ref: ArtifactRef
    automatic_search_result_ref: ArtifactRef

    @model_validator(mode="after")
    def exact_run_refs(self) -> Self:
        expected = (
            (self.run_closure_ref, STUDY_RUN_CLOSURE_MEDIA_TYPE, "run_closure_ref"),
            (self.search_run_ref, SEARCH_RUN_MANIFEST_MEDIA_TYPE, "search_run_ref"),
            (
                self.automatic_search_result_ref,
                AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE,
                "automatic_search_result_ref",
            ),
        )
        for ref, media_type, field_name in expected:
            if ref.media_type != media_type:
                raise ValueError(f"{field_name} declares the wrong media type")
        return self


class ReplayStudyGateResult(ImmutableModel):
    """Upper-level, replayable boundary for the fixture's accepted GATE evidence."""

    schema_version: Literal["1"] = "1"
    study_result_ref: ArtifactRef
    baseline_gate_acceptance_ref: ArtifactRef
    baseline_gate_closure_ref: ArtifactRef
    study_controller_manifest_ref: ArtifactRef
    baseline_study_plan_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    plan_fingerprint: Sha256
    barrier_closed_tail_ref: ArtifactRef
    barrier_closure_ref: ArtifactRef
    sealed_authorized_tail_ref: ArtifactRef
    sealed_authorization_ref: ArtifactRef
    runs: Annotated[tuple[ReplayStudyGateRun, ...], Field(min_length=8, max_length=8)]
    exact_run_count: Literal[8] = 8
    accepted_for: Literal["non-reportable-replay-study-gate-result"] = (
        "non-reportable-replay-study-gate-result"
    )
    non_reportable_fixture: Literal[True] = True
    reportable_benchmark_result: Literal[False] = False
    limitation: NonEmptyStr = (
        "post-barrier replay fixture packaging of trusted GATE evidence; "
        "not a sealed benchmark score"
    )

    @model_validator(mode="after")
    def exact_result_refs(self) -> Self:
        expected = (
            (self.study_result_ref, REPLAY_STUDY_RESULT_MEDIA_TYPE, "study_result_ref"),
            (
                self.baseline_gate_acceptance_ref,
                BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE,
                "baseline_gate_acceptance_ref",
            ),
            (
                self.baseline_gate_closure_ref,
                BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE,
                "baseline_gate_closure_ref",
            ),
            (
                self.study_controller_manifest_ref,
                STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
                "study_controller_manifest_ref",
            ),
            (
                self.baseline_study_plan_ref,
                BASELINE_STUDY_PLAN_MEDIA_TYPE,
                "baseline_study_plan_ref",
            ),
            (
                self.barrier_closed_tail_ref,
                STUDY_JOURNAL_EVENT_MEDIA_TYPE,
                "barrier_closed_tail_ref",
            ),
            (
                self.barrier_closure_ref,
                STUDY_BARRIER_CLOSURE_MEDIA_TYPE,
                "barrier_closure_ref",
            ),
            (
                self.sealed_authorized_tail_ref,
                STUDY_JOURNAL_EVENT_MEDIA_TYPE,
                "sealed_authorized_tail_ref",
            ),
            (
                self.sealed_authorization_ref,
                STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE,
                "sealed_authorization_ref",
            ),
        )
        for ref, media_type, field_name in expected:
            if ref.media_type != media_type:
                raise ValueError(f"{field_name} declares the wrong media type")
        _require_json_ref(self.analysis_plan_ref, field_name="analysis_plan_ref")
        coordinates = tuple(_run_coordinate(run.key) for run in self.runs)
        expected_coordinates = tuple(
            sorted((kind.value, seed) for kind in BaselineKind for seed in SEARCH_RUN_SEEDS)
        )
        if coordinates != expected_coordinates:
            raise ValueError("replay study gate result does not cover the four-by-two schedule")
        _require_unique_refs(
            tuple(run.run_closure_ref for run in self.runs),
            "runs.run_closure_ref",
        )
        _require_unique_refs(
            tuple(run.search_run_ref for run in self.runs),
            "runs.search_run_ref",
        )
        _require_unique_refs(
            tuple(run.automatic_search_result_ref for run in self.runs),
            "runs.automatic_search_result_ref",
        )
        return self


def publish_replay_study_gate_result(
    repository: ArtifactRepository,
    *,
    study_result_ref: ArtifactRef,
    baseline_gate_acceptance_ref: ArtifactRef,
    event_verifier: StudyEventVerificationCapability,
) -> ArtifactRef:
    """Publish and immediately replay-check a non-reportable replay GATE result."""

    result = build_replay_study_gate_result(
        repository,
        study_result_ref=study_result_ref,
        baseline_gate_acceptance_ref=baseline_gate_acceptance_ref,
        event_verifier=event_verifier,
    )
    ref = repository.put_json(result, media_type=REPLAY_STUDY_GATE_RESULT_MEDIA_TYPE)
    verified = verify_replay_study_gate_result(
        repository,
        ref,
        event_verifier=event_verifier,
    )
    if verified != result:
        raise ReplayStudyGateResultError("published replay study GATE result changed content")
    return ArtifactRef.model_validate(ref, strict=True)


def build_replay_study_gate_result(
    repository: ArtifactRepository,
    *,
    study_result_ref: ArtifactRef,
    baseline_gate_acceptance_ref: ArtifactRef,
    event_verifier: StudyEventVerificationCapability,
) -> ReplayStudyGateResult:
    """Build the upper-level replay result without trusting caller-authored joins."""

    checked_repository = _require_repository(repository)
    checked_result_ref = _validate_ref(
        study_result_ref,
        REPLAY_STUDY_RESULT_MEDIA_TYPE,
        "replay study result",
    )
    study_result = _load_model(
        checked_repository,
        checked_result_ref,
        ReplayStudyResult,
        "replay study result",
    )
    try:
        acceptance = verify_baseline_gate_study_acceptance(
            checked_repository,
            baseline_gate_acceptance_ref,
            event_verifier=event_verifier,
        )
    except BaselineGateAcceptanceError as exc:
        raise ReplayStudyGateResultError(str(exc)) from exc
    authorization = _load_model(
        checked_repository,
        study_result.sealed_authorization_ref,
        StudySealedAuthorization,
        "study sealed authorization",
    )
    _join_control_boundary(
        study_result=study_result,
        acceptance_ref=baseline_gate_acceptance_ref,
        acceptance=acceptance,
        authorization=authorization,
    )
    runs = _joined_runs(
        checked_repository,
        study_result=study_result,
        authorization=authorization,
        accepted_run_closure_refs=acceptance.expected_run_closure_refs,
    )
    return ReplayStudyGateResult(
        study_result_ref=checked_result_ref,
        baseline_gate_acceptance_ref=_validate_ref(
            baseline_gate_acceptance_ref,
            BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE,
            "baseline gate study acceptance",
        ),
        baseline_gate_closure_ref=acceptance.baseline_gate_closure_ref,
        study_controller_manifest_ref=study_result.study_controller_manifest_ref,
        baseline_study_plan_ref=study_result.baseline_study_plan_ref,
        analysis_plan_ref=study_result.analysis_plan_ref,
        plan_fingerprint=acceptance.plan_fingerprint,
        barrier_closed_tail_ref=study_result.barrier_closed_tail_ref,
        barrier_closure_ref=acceptance.barrier_closure_ref,
        sealed_authorized_tail_ref=study_result.sealed_authorized_tail_ref,
        sealed_authorization_ref=study_result.sealed_authorization_ref,
        runs=runs,
    )


def verify_replay_study_gate_result(
    repository: ArtifactRepository,
    result_ref: ArtifactRef,
    *,
    event_verifier: StudyEventVerificationCapability,
) -> ReplayStudyGateResult:
    """Reload a replay GATE result and rebuild its study/acceptance joins."""

    checked_repository = _require_repository(repository)
    checked_ref = _validate_ref(
        result_ref,
        REPLAY_STUDY_GATE_RESULT_MEDIA_TYPE,
        "replay study gate result",
    )
    result = _load_model(
        checked_repository,
        checked_ref,
        ReplayStudyGateResult,
        "replay study gate result",
    )
    rebuilt = build_replay_study_gate_result(
        checked_repository,
        study_result_ref=result.study_result_ref,
        baseline_gate_acceptance_ref=result.baseline_gate_acceptance_ref,
        event_verifier=event_verifier,
    )
    if rebuilt != result:
        raise ReplayStudyGateResultError("replay study GATE result cannot be replayed")
    return result


def _join_control_boundary(
    *,
    study_result: ReplayStudyResult,
    acceptance_ref: ArtifactRef,
    acceptance: BaselineGateStudyAcceptance,
    authorization: StudySealedAuthorization,
) -> None:
    expected = (
        study_result.study_controller_manifest_ref,
        study_result.baseline_study_plan_ref,
        study_result.analysis_plan_ref,
        study_result.barrier_closed_tail_ref,
        study_result.sealed_authorized_tail_ref,
        study_result.sealed_authorization_ref,
    )
    actual = (
        acceptance.study_controller_manifest_ref,
        acceptance.baseline_study_plan_ref,
        authorization.analysis_plan_ref,
        authorization.barrier_closed_tail_ref,
        acceptance.sealed_authorized_tail_ref,
        acceptance.sealed_authorization_ref,
    )
    if actual != expected:
        raise ReplayStudyGateResultError(
            "accepted GATE evidence does not belong to the replay study result"
        )
    _validate_ref(
        acceptance_ref,
        BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE,
        "baseline gate study acceptance",
    )
    if acceptance.barrier_closure_ref != authorization.barrier_closure_ref:
        raise ReplayStudyGateResultError(
            "accepted GATE evidence points at another study barrier closure"
        )
    if acceptance.plan_fingerprint != authorization.plan_fingerprint:
        raise ReplayStudyGateResultError("accepted GATE evidence uses another study plan")
    if acceptance.expected_run_closure_refs != tuple(
        binding.run_closure_ref for binding in authorization.bindings
    ):
        raise ReplayStudyGateResultError("accepted GATE evidence omits study run closures")


def _joined_runs(
    repository: ArtifactRepository,
    *,
    study_result: ReplayStudyResult,
    authorization: StudySealedAuthorization,
    accepted_run_closure_refs: tuple[ArtifactRef, ...],
) -> tuple[ReplayStudyGateRun, ...]:
    result_by_key = {_run_coordinate(run.key): run for run in study_result.runs}
    authorized_by_key = {
        _run_coordinate(binding.key): binding for binding in authorization.bindings
    }
    if len(result_by_key) != len(study_result.runs):
        raise ReplayStudyGateResultError("replay study result contains duplicate run keys")
    if result_by_key.keys() != authorized_by_key.keys():
        raise ReplayStudyGateResultError(
            "replay study result and sealed authorization cover different runs"
        )
    if set(accepted_run_closure_refs) != {
        binding.run_closure_ref for binding in authorization.bindings
    }:
        raise ReplayStudyGateResultError("accepted run closures differ from authorization")

    runs: list[ReplayStudyGateRun] = []
    for coordinate in sorted(result_by_key):
        run = result_by_key[coordinate]
        binding = authorized_by_key[coordinate]
        closure = _load_model(repository, binding.run_closure_ref, StudyRunClosure, "run closure")
        _join_run_boundary(study_result=study_result, run=run, closure=closure)
        runs.append(
            ReplayStudyGateRun(
                key=run.key,
                run_closure_ref=binding.run_closure_ref,
                search_run_ref=run.search_run_ref,
                automatic_search_result_ref=run.automatic_search_result_ref,
            )
        )
    return tuple(runs)


def _join_run_boundary(
    *,
    study_result: ReplayStudyResult,
    run: ReplayStudyRunResult,
    closure: StudyRunClosure,
) -> None:
    expected = (
        study_result.study_controller_manifest_ref,
        study_result.baseline_study_plan_ref,
        study_result.analysis_plan_ref,
        run.key,
        run.search_run_ref,
        run.search_controller_manifest_ref,
        run.experiment_ref,
        run.automatic_search_result_ref,
        run.final_search_tail_ref,
        run.final_experiment_tail_ref,
    )
    actual = (
        closure.study_controller_manifest_ref,
        closure.baseline_study_plan_ref,
        closure.analysis_plan_ref,
        closure.key,
        closure.search_run_ref,
        closure.search_controller_manifest_ref,
        closure.experiment_ref,
        closure.automatic_search_result_ref,
        closure.search_selection_tail_ref,
        closure.experiment_selection_tail_ref,
    )
    if actual != expected:
        raise ReplayStudyGateResultError("replay study run differs from its study closure")


def _run_coordinate(key: StudyRunKey) -> tuple[str, int]:
    return key.baseline_kind.value, key.search_run_seed


def _load_model[ModelT](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    try:
        return repository.get_json(ref, model_type)
    except Exception as exc:
        raise ReplayStudyGateResultError(f"{label} cannot be loaded") from exc


def _require_repository(repository: ArtifactRepository) -> ArtifactRepository:
    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    return repository


def _validate_ref(ref: ArtifactRef, expected_media_type: str, label: str) -> ArtifactRef:
    checked = ArtifactRef.model_validate(ref, strict=True)
    if checked.media_type != expected_media_type:
        raise ReplayStudyGateResultError(f"{label} declares the wrong media type")
    return checked


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


def _require_unique_refs(refs: tuple[ArtifactRef, ...], field_name: str) -> None:
    if len({ref.sha256 for ref in refs}) != len(refs):
        raise ValueError(f"{field_name} must not contain duplicate refs")


__all__ = [
    "REPLAY_STUDY_GATE_RESULT_MEDIA_TYPE",
    "ReplayStudyGateResult",
    "ReplayStudyGateResultError",
    "ReplayStudyGateRun",
    "build_replay_study_gate_result",
    "publish_replay_study_gate_result",
    "verify_replay_study_gate_result",
]
