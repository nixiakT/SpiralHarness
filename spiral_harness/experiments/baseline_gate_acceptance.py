"""Accept baseline GATE closures behind a study sealed barrier."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from spiral_harness.core.experiment import (
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    ExperimentManifest,
    ProtocolManifest,
    ProtocolPartition,
)
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.experiments.baseline_gate_closure import (
    BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE,
    BaselineGateClosureError,
    BaselineGateStudyClosure,
    verify_baseline_gate_study_closure,
)
from spiral_harness.experiments.baselines import (
    BASELINE_STUDY_PLAN_MEDIA_TYPE,
    BaselineStudyPlan,
)
from spiral_harness.experiments.study import (
    STUDY_BARRIER_CLOSURE_MEDIA_TYPE,
    STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
    STUDY_JOURNAL_EVENT_MEDIA_TYPE,
    STUDY_RUN_CLOSURE_MEDIA_TYPE,
    STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE,
    StudyBarrierClosure,
    StudyEventVerificationCapability,
    StudyJournal,
    StudyRunClosure,
    StudySealedAuthorization,
    StudyState,
)
from spiral_harness.storage.protocol import ArtifactRepository

BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.baseline-gate-study-acceptance.v1+json"
)


class BaselineGateAcceptanceError(RuntimeError):
    """Raised when a baseline GATE closure cannot be accepted for a study."""


class BaselineGateStudyAcceptance(ImmutableModel):
    """Replayable evidence that a GATE closure passed the study barrier."""

    schema_version: Literal["1"] = "1"
    study_controller_manifest_ref: ArtifactRef
    baseline_study_plan_ref: ArtifactRef
    plan_fingerprint: Sha256
    sealed_authorized_tail_ref: ArtifactRef
    sealed_authorization_ref: ArtifactRef
    barrier_closure_ref: ArtifactRef
    baseline_gate_closure_ref: ArtifactRef
    protocol_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    seed_harness_ref: ArtifactRef
    expected_run_closure_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    accepted_for: Literal["post-study-barrier-gate-closure"] = "post-study-barrier-gate-closure"
    reportable_benchmark_result: Literal[False] = False
    limitation: NonEmptyStr = (
        "structural acceptance of trusted GATE evidence; sealed final score is separate"
    )

    @model_validator(mode="after")
    def exact_acceptance_refs(self) -> BaselineGateStudyAcceptance:
        expected = (
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
                self.sealed_authorized_tail_ref,
                STUDY_JOURNAL_EVENT_MEDIA_TYPE,
                "sealed_authorized_tail_ref",
            ),
            (
                self.sealed_authorization_ref,
                STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE,
                "sealed_authorization_ref",
            ),
            (
                self.barrier_closure_ref,
                STUDY_BARRIER_CLOSURE_MEDIA_TYPE,
                "barrier_closure_ref",
            ),
            (
                self.baseline_gate_closure_ref,
                BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE,
                "baseline_gate_closure_ref",
            ),
            (
                self.protocol_ref,
                PROTOCOL_MANIFEST_MEDIA_TYPE,
                "protocol_ref",
            ),
            (
                self.seed_harness_ref,
                HARNESS_MANIFEST_MEDIA_TYPE,
                "seed_harness_ref",
            ),
        )
        for ref, media_type, field_name in expected:
            if ref.media_type != media_type:
                raise ValueError(f"{field_name} declares the wrong media type")
        _require_json_ref(self.gate_split_ref, field_name="gate_split_ref")
        _require_ref_media(
            self.expected_run_closure_refs,
            STUDY_RUN_CLOSURE_MEDIA_TYPE,
            "expected_run_closure_refs",
        )
        return self


def publish_baseline_gate_study_acceptance(
    repository: ArtifactRepository,
    *,
    study_controller_manifest_ref: ArtifactRef,
    sealed_authorized_tail_ref: ArtifactRef,
    event_verifier: StudyEventVerificationCapability,
    baseline_gate_closure_ref: ArtifactRef,
) -> ArtifactRef:
    """Publish a verified acceptance artifact for one study-level GATE closure."""

    acceptance = build_baseline_gate_study_acceptance(
        repository,
        study_controller_manifest_ref=study_controller_manifest_ref,
        sealed_authorized_tail_ref=sealed_authorized_tail_ref,
        event_verifier=event_verifier,
        baseline_gate_closure_ref=baseline_gate_closure_ref,
    )
    ref = repository.put_json(acceptance, media_type=BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE)
    verified = verify_baseline_gate_study_acceptance(
        repository,
        ref,
        event_verifier=event_verifier,
    )
    if verified != acceptance:
        raise BaselineGateAcceptanceError("published baseline GATE acceptance changed content")
    return ArtifactRef.model_validate(ref, strict=True)


def build_baseline_gate_study_acceptance(
    repository: ArtifactRepository,
    *,
    study_controller_manifest_ref: ArtifactRef,
    sealed_authorized_tail_ref: ArtifactRef,
    event_verifier: StudyEventVerificationCapability,
    baseline_gate_closure_ref: ArtifactRef,
) -> BaselineGateStudyAcceptance:
    """Verify and build acceptance without trusting caller-authored summaries."""

    checked_repository = _require_repository(repository)
    journal = _study_journal(
        checked_repository,
        study_controller_manifest_ref=study_controller_manifest_ref,
        event_verifier=event_verifier,
    )
    sealed_tail = _validate_ref(
        sealed_authorized_tail_ref,
        STUDY_JOURNAL_EVENT_MEDIA_TYPE,
        "sealed authorized tail",
    )
    snapshot = journal.replay(sealed_tail)
    if snapshot.state is not StudyState.SEALED_AUTHORIZED:
        raise BaselineGateAcceptanceError("study tail is not SEALED_AUTHORIZED")
    if snapshot.sealed_authorization_ref is None or snapshot.barrier_closure_ref is None:
        raise BaselineGateAcceptanceError("study sealed tail is missing closure evidence")
    authorization = checked_repository.get_json(
        snapshot.sealed_authorization_ref,
        StudySealedAuthorization,
    )
    barrier = checked_repository.get_json(snapshot.barrier_closure_ref, StudyBarrierClosure)
    plan = checked_repository.get_json(journal.manifest.baseline_study_plan_ref, BaselineStudyPlan)
    gate_closure = _verified_gate_closure(
        checked_repository,
        baseline_gate_closure_ref,
        plan=plan,
    )
    protocol_ref, gate_split_ref, seed_harness_ref = _expected_study_execution_boundary(
        checked_repository,
        barrier,
    )
    _join_acceptance_boundary(
        gate_closure=gate_closure,
        protocol_ref=protocol_ref,
        gate_split_ref=gate_split_ref,
        seed_harness_ref=seed_harness_ref,
    )
    return BaselineGateStudyAcceptance(
        study_controller_manifest_ref=journal.study_controller_manifest_ref,
        baseline_study_plan_ref=journal.manifest.baseline_study_plan_ref,
        plan_fingerprint=journal.plan.fingerprint,
        sealed_authorized_tail_ref=sealed_tail,
        sealed_authorization_ref=snapshot.sealed_authorization_ref,
        barrier_closure_ref=snapshot.barrier_closure_ref,
        baseline_gate_closure_ref=_validate_ref(
            baseline_gate_closure_ref,
            BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE,
            "baseline gate study closure",
        ),
        protocol_ref=protocol_ref,
        gate_split_ref=gate_split_ref,
        seed_harness_ref=seed_harness_ref,
        expected_run_closure_refs=tuple(
            binding.run_closure_ref for binding in authorization.bindings
        ),
    )


def verify_baseline_gate_study_acceptance(
    repository: ArtifactRepository,
    acceptance_ref: ArtifactRef,
    *,
    event_verifier: StudyEventVerificationCapability,
) -> BaselineGateStudyAcceptance:
    """Reload an acceptance artifact and replay every bound study/runner proof."""

    checked_repository = _require_repository(repository)
    checked_ref = _validate_ref(
        acceptance_ref,
        BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE,
        "baseline gate study acceptance",
    )
    acceptance = _load_acceptance(checked_repository, checked_ref)
    rebuilt = build_baseline_gate_study_acceptance(
        checked_repository,
        study_controller_manifest_ref=acceptance.study_controller_manifest_ref,
        sealed_authorized_tail_ref=acceptance.sealed_authorized_tail_ref,
        event_verifier=event_verifier,
        baseline_gate_closure_ref=acceptance.baseline_gate_closure_ref,
    )
    if rebuilt != acceptance:
        raise BaselineGateAcceptanceError("baseline GATE acceptance cannot be replayed")
    return acceptance


def _expected_study_execution_boundary(
    repository: ArtifactRepository,
    barrier: StudyBarrierClosure,
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef]:
    if not barrier.bindings:
        raise BaselineGateAcceptanceError("study barrier contains no run bindings")
    protocol_refs: set[ArtifactRef] = set()
    seed_harness_refs: set[ArtifactRef] = set()
    for binding in barrier.bindings:
        run_closure = repository.get_json(binding.run_closure_ref, StudyRunClosure)
        experiment = repository.get_json(run_closure.experiment_ref, ExperimentManifest)
        protocol_refs.add(experiment.protocol_ref)
        seed_harness_refs.add(experiment.seed_harness_ref)
    if len(protocol_refs) != 1:
        raise BaselineGateAcceptanceError("study run closures use multiple protocols")
    if len(seed_harness_refs) != 1:
        raise BaselineGateAcceptanceError("study run closures use multiple seed harnesses")
    protocol_ref = next(iter(protocol_refs))
    protocol = repository.get_json(protocol_ref, ProtocolManifest)
    gate_split_refs = tuple(
        split.manifest_ref for split in protocol.splits if split.partition is ProtocolPartition.GATE
    )
    if len(gate_split_refs) != 1:
        raise BaselineGateAcceptanceError("study protocol does not freeze one GATE split")
    return protocol_ref, gate_split_refs[0], next(iter(seed_harness_refs))


def _join_acceptance_boundary(
    *,
    gate_closure: BaselineGateStudyClosure,
    protocol_ref: ArtifactRef,
    gate_split_ref: ArtifactRef,
    seed_harness_ref: ArtifactRef,
) -> None:
    expected = (protocol_ref, gate_split_ref, seed_harness_ref)
    actual = (
        gate_closure.protocol_ref,
        gate_closure.gate_split_ref,
        gate_closure.parent_harness_ref,
    )
    if actual != expected:
        raise BaselineGateAcceptanceError(
            "baseline GATE closure differs from the study execution boundary"
        )
    if gate_closure.reportable_benchmark_result:
        raise BaselineGateAcceptanceError("baseline GATE closure must remain non-reportable")


def _study_journal(
    repository: ArtifactRepository,
    *,
    study_controller_manifest_ref: ArtifactRef,
    event_verifier: StudyEventVerificationCapability,
) -> StudyJournal:
    if type(event_verifier) is not StudyEventVerificationCapability:
        raise TypeError("event_verifier must be an exact StudyEventVerificationCapability")
    try:
        return StudyJournal(
            repository,
            study_controller_manifest_ref=_validate_ref(
                study_controller_manifest_ref,
                STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
                "study controller manifest",
            ),
            event_verifier=event_verifier,
        )
    except Exception as exc:
        raise BaselineGateAcceptanceError(f"study journal cannot be verified: {exc}") from exc


def _verified_gate_closure(
    repository: ArtifactRepository,
    closure_ref: ArtifactRef,
    *,
    plan: BaselineStudyPlan,
) -> BaselineGateStudyClosure:
    try:
        return verify_baseline_gate_study_closure(repository, closure_ref, plan=plan)
    except BaselineGateClosureError as exc:
        raise BaselineGateAcceptanceError(str(exc)) from exc


def _load_acceptance(
    repository: ArtifactRepository,
    acceptance_ref: ArtifactRef,
) -> BaselineGateStudyAcceptance:
    try:
        return repository.get_json(acceptance_ref, BaselineGateStudyAcceptance)
    except Exception as exc:
        raise BaselineGateAcceptanceError("baseline GATE acceptance cannot be loaded") from exc


def _require_repository(repository: ArtifactRepository) -> ArtifactRepository:
    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    return repository


def _validate_ref(ref: ArtifactRef, expected_media_type: str, label: str) -> ArtifactRef:
    checked = ArtifactRef.model_validate(ref, strict=True)
    if checked.media_type != expected_media_type:
        raise BaselineGateAcceptanceError(f"{label} declares the wrong media type")
    return checked


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


def _require_ref_media(
    refs: tuple[ArtifactRef, ...],
    expected_media_type: str,
    field_name: str,
) -> None:
    if len({ref.sha256 for ref in refs}) != len(refs):
        raise ValueError(f"{field_name} must not contain duplicates")
    for ref in refs:
        if ref.media_type != expected_media_type:
            raise ValueError(f"{field_name} contains the wrong media type")


__all__ = [
    "BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE",
    "BaselineGateAcceptanceError",
    "BaselineGateStudyAcceptance",
    "build_baseline_gate_study_acceptance",
    "publish_baseline_gate_study_acceptance",
    "verify_baseline_gate_study_acceptance",
]
