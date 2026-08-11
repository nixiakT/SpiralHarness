"""Attested four-arm study barrier over independently closed search runs.

The study controller never accepts caller-authored aggregate usage reports.
Instead, every expected ``(baseline kind, search-run seed)`` cell must present
two live capabilities: its :class:`SearchController` re-authenticates the
current search selection tail, and its underlying :class:`ExperimentController`
re-authenticates the current experiment selection tail.  A controller-authored
run closure binds both before the study can cross its all-runs barrier.

This barrier detects and rejects a run controller that entered sealed state
early; it does not revoke the lower-level ``ExperimentController.start_sealed``
API or claim to prevent an out-of-band caller from observing that split.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.experiment import EXPERIMENT_MANIFEST_MEDIA_TYPE, ExperimentManifest
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.evolution.models import (
    SearchPolicy,
    SearchRunManifest,
    SearchStoppingPolicy,
    StrategyPluginManifest,
)
from spiral_harness.evolution.orchestrator import (
    AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE,
    SEARCH_RUN_ADMISSION_REPORT_MEDIA_TYPE,
    AutomaticSearchLoop,
    AutomaticSearchLoopResult,
    SearchRunAdmissionExpectation,
    SearchRunAdmissionService,
)
from spiral_harness.experiments.baselines import (
    BASELINE_STUDY_PLAN_MEDIA_TYPE,
    BaselineKind,
    BaselineStudyPlan,
)
from spiral_harness.experiments.controller import ExperimentController
from spiral_harness.experiments.lifecycle import (
    EXPERIMENT_JOURNAL_ENTRY_MEDIA_TYPE,
    SELECTION_CLOSURE_MEDIA_TYPE,
    ExperimentJournal,
    ExperimentState,
    SelectionClosure,
)
from spiral_harness.experiments.search import (
    SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
    SEARCH_JOURNAL_EVENT_MEDIA_TYPE,
    SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    SEARCH_SELECTION_CLOSURE_MEDIA_TYPE,
    SearchController,
    SearchControllerManifest,
    SearchSelectionClosure,
    SearchState,
)
from spiral_harness.storage.protocol import ArtifactRepository

STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE = (
    "application/vnd.spiral-harness.study-controller-manifest.v1+json"
)
STUDY_JOURNAL_EVENT_MEDIA_TYPE = "application/vnd.spiral-harness.study-journal-event.v1+json"
STUDY_RUN_CLOSURE_MEDIA_TYPE = "application/vnd.spiral-harness.study-run-closure.v1+json"
STUDY_BARRIER_CLOSURE_MEDIA_TYPE = "application/vnd.spiral-harness.study-barrier-closure.v1+json"
STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.study-sealed-authorization.v1+json"
)
_EVENT_ATTESTATION_DOMAIN = b"spiral-harness:study-journal-event-attestation:v1\x00"
_EVENT_ATTESTOR_ID_DOMAIN = b"spiral-harness:study-journal-event-attestor-id:v1\x00"


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


class StudyControllerError(RuntimeError):
    """Base error for fail-closed study orchestration."""


class StudyJournalIntegrityError(StudyControllerError):
    """Raised when a linked study journal or one of its closures is invalid."""


class StudyJournalCycleError(StudyJournalIntegrityError):
    """Raised when a study journal link revisits a tail."""


class StudyEventAttestationError(StudyJournalIntegrityError):
    """Raised when a journal event is not authentic for the frozen signer."""


class StaleStudyTailError(StudyControllerError):
    """Raised when a caller mutates an old or foreign study branch."""


class StudyBarrierError(StudyControllerError):
    """Raised when registration or sealing violates the complete-run barrier."""


class StudyState(StrEnum):
    """Persisted states of one exact baseline study."""

    FROZEN = "frozen"
    COLLECTING = "collecting"
    BARRIER_CLOSED = "barrier_closed"
    SEALED_AUTHORIZED = "sealed_authorized"
    INVALIDATED = "invalidated"


class StudyEventKind(StrEnum):
    """Operations represented by attested study events."""

    FROZEN = "frozen"
    COLLECTION_STARTED = "collection_started"
    RUN_REGISTERED = "run_registered"
    BARRIER_CLOSED = "barrier_closed"
    SEALED_AUTHORIZED = "sealed_authorized"
    INVALIDATED = "invalidated"


class StudyRunKey(ImmutableModel):
    """One required cell in the four-arm independent-search schedule."""

    baseline_kind: BaselineKind
    search_run_seed: Annotated[int, Field(ge=0, strict=True)]


def _key_order(key: StudyRunKey) -> tuple[str, int]:
    return key.baseline_kind.value, key.search_run_seed


def _canonical_keys(keys: tuple[StudyRunKey, ...]) -> tuple[StudyRunKey, ...]:
    ordered = tuple(sorted(keys, key=_key_order))
    if len(ordered) != len(set(_key_order(key) for key in ordered)):
        raise ValueError("study run keys must not contain duplicates")
    return ordered


def expected_run_keys(plan: BaselineStudyPlan) -> tuple[StudyRunKey, ...]:
    """Derive, never accept, the complete ``4 x search_run_seeds`` key set."""

    validated = BaselineStudyPlan.model_validate(plan)
    return _canonical_keys(
        tuple(
            StudyRunKey(baseline_kind=arm.kind, search_run_seed=seed)
            for arm in validated.arms
            for seed in arm.evaluation.search_run_seeds
        )
    )


class StudyExpectedRunManifest(ImmutableModel):
    """One pre-registered run manifest for an exact plan-derived study cell."""

    key: StudyRunKey
    search_run_ref: ArtifactRef

    @model_validator(mode="after")
    def exact_run_manifest_type(self) -> StudyExpectedRunManifest:
        if self.search_run_ref.media_type != SEARCH_RUN_MANIFEST_MEDIA_TYPE:
            raise ValueError("search_run_ref declares the wrong media type")
        return self


def _canonical_expected_runs(
    runs: tuple[StudyExpectedRunManifest, ...],
) -> tuple[StudyExpectedRunManifest, ...]:
    ordered = tuple(sorted(runs, key=lambda run: _key_order(run.key)))
    if len(ordered) != len(set(_key_order(run.key) for run in ordered)):
        raise ValueError("expected run manifests must not contain duplicate keys")
    run_hashes = tuple(run.search_run_ref.sha256 for run in ordered)
    if len(run_hashes) != len(set(run_hashes)):
        raise ValueError("one search-run manifest cannot satisfy multiple study keys")
    return ordered


class StudyControllerManifest(ImmutableModel):
    """Freeze the exact baseline plan and study event signer."""

    schema_version: Literal["1"] = "1"
    baseline_study_plan_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    expected_run_manifests: Annotated[tuple[StudyExpectedRunManifest, ...], Field(min_length=1)]
    study_id: NonEmptyStr
    journal_attestor_id: Sha256

    @field_validator("expected_run_manifests")
    @classmethod
    def canonicalize_expected_run_manifests(
        cls,
        runs: tuple[StudyExpectedRunManifest, ...],
    ) -> tuple[StudyExpectedRunManifest, ...]:
        return _canonical_expected_runs(runs)

    @model_validator(mode="after")
    def exact_plan_media_type(self) -> StudyControllerManifest:
        if self.baseline_study_plan_ref.media_type != BASELINE_STUDY_PLAN_MEDIA_TYPE:
            raise ValueError("baseline_study_plan_ref declares the wrong media type")
        _require_json_ref(self.analysis_plan_ref, field_name="analysis_plan_ref")
        return self


class StudyRunClosure(ImmutableModel):
    """Controller-authored join of one search and experiment selection head."""

    schema_version: Literal["1"] = "1"
    study_controller_manifest_ref: ArtifactRef
    baseline_study_plan_ref: ArtifactRef
    plan_fingerprint: Sha256
    key: StudyRunKey
    search_controller_manifest_ref: ArtifactRef
    search_run_ref: ArtifactRef
    search_run_admission_report_ref: ArtifactRef
    automatic_search_result_ref: ArtifactRef
    search_selection_tail_ref: ArtifactRef
    search_selection_closure_ref: ArtifactRef
    experiment_ref: ArtifactRef
    experiment_selection_tail_ref: ArtifactRef
    experiment_selection_closure_ref: ArtifactRef
    champion_harness_ref: ArtifactRef
    champion_candidate_ref: ArtifactRef | None
    analysis_plan_ref: ArtifactRef
    usage_tail_ref: ArtifactRef | None

    @model_validator(mode="after")
    def exact_reference_types(self) -> StudyRunClosure:
        exact = (
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
                self.search_controller_manifest_ref,
                SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
                "search_controller_manifest_ref",
            ),
            (self.search_run_ref, SEARCH_RUN_MANIFEST_MEDIA_TYPE, "search_run_ref"),
            (
                self.search_run_admission_report_ref,
                SEARCH_RUN_ADMISSION_REPORT_MEDIA_TYPE,
                "search_run_admission_report_ref",
            ),
            (
                self.automatic_search_result_ref,
                AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE,
                "automatic_search_result_ref",
            ),
            (
                self.search_selection_tail_ref,
                SEARCH_JOURNAL_EVENT_MEDIA_TYPE,
                "search_selection_tail_ref",
            ),
            (
                self.search_selection_closure_ref,
                SEARCH_SELECTION_CLOSURE_MEDIA_TYPE,
                "search_selection_closure_ref",
            ),
            (self.experiment_ref, EXPERIMENT_MANIFEST_MEDIA_TYPE, "experiment_ref"),
            (
                self.experiment_selection_tail_ref,
                EXPERIMENT_JOURNAL_ENTRY_MEDIA_TYPE,
                "experiment_selection_tail_ref",
            ),
            (
                self.experiment_selection_closure_ref,
                SELECTION_CLOSURE_MEDIA_TYPE,
                "experiment_selection_closure_ref",
            ),
        )
        for ref, expected, field_name in exact:
            if ref.media_type != expected:
                raise ValueError(f"{field_name} declares the wrong media type")
        _require_json_ref(self.analysis_plan_ref, field_name="analysis_plan_ref")
        return self


class StudyRunBinding(ImmutableModel):
    """Canonical mapping from one expected key to its trusted closure."""

    key: StudyRunKey
    run_closure_ref: ArtifactRef

    @model_validator(mode="after")
    def exact_closure_type(self) -> StudyRunBinding:
        if self.run_closure_ref.media_type != STUDY_RUN_CLOSURE_MEDIA_TYPE:
            raise ValueError("run_closure_ref declares the wrong media type")
        return self


def _canonical_bindings(
    bindings: tuple[StudyRunBinding, ...],
) -> tuple[StudyRunBinding, ...]:
    ordered = tuple(sorted(bindings, key=lambda binding: _key_order(binding.key)))
    if len(ordered) != len(set(_key_order(binding.key) for binding in ordered)):
        raise ValueError("study run bindings must not contain duplicate keys")
    closure_hashes = [binding.run_closure_ref.sha256 for binding in ordered]
    if len(closure_hashes) != len(set(closure_hashes)):
        raise ValueError("one run closure cannot satisfy multiple study keys")
    return ordered


class StudyBarrierClosure(ImmutableModel):
    """Immutable proof that every plan-derived run key was registered."""

    schema_version: Literal["1"] = "1"
    study_controller_manifest_ref: ArtifactRef
    baseline_study_plan_ref: ArtifactRef
    plan_fingerprint: Sha256
    analysis_plan_ref: ArtifactRef
    collection_tail_ref: ArtifactRef
    expected_keys: Annotated[tuple[StudyRunKey, ...], Field(min_length=1)]
    bindings: Annotated[tuple[StudyRunBinding, ...], Field(min_length=1)]

    @field_validator("expected_keys")
    @classmethod
    def canonicalize_expected_keys(cls, keys: tuple[StudyRunKey, ...]) -> tuple[StudyRunKey, ...]:
        return _canonical_keys(keys)

    @field_validator("bindings")
    @classmethod
    def canonicalize_bindings(
        cls, bindings: tuple[StudyRunBinding, ...]
    ) -> tuple[StudyRunBinding, ...]:
        return _canonical_bindings(bindings)

    @model_validator(mode="after")
    def exact_types_and_complete_mapping(self) -> StudyBarrierClosure:
        if self.study_controller_manifest_ref.media_type != STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE:
            raise ValueError("study_controller_manifest_ref declares the wrong media type")
        if self.baseline_study_plan_ref.media_type != BASELINE_STUDY_PLAN_MEDIA_TYPE:
            raise ValueError("baseline_study_plan_ref declares the wrong media type")
        if self.collection_tail_ref.media_type != STUDY_JOURNAL_EVENT_MEDIA_TYPE:
            raise ValueError("collection_tail_ref declares the wrong media type")
        _require_json_ref(self.analysis_plan_ref, field_name="analysis_plan_ref")
        if tuple(binding.key for binding in self.bindings) != self.expected_keys:
            raise ValueError("barrier bindings must cover exactly every expected key")
        return self


class StudySealedAuthorization(ImmutableModel):
    """Study-level sealed access authorization derived only after the barrier."""

    schema_version: Literal["1"] = "1"
    study_controller_manifest_ref: ArtifactRef
    baseline_study_plan_ref: ArtifactRef
    plan_fingerprint: Sha256
    analysis_plan_ref: ArtifactRef
    barrier_closed_tail_ref: ArtifactRef
    barrier_closure_ref: ArtifactRef
    bindings: Annotated[tuple[StudyRunBinding, ...], Field(min_length=1)]
    sealed_access_authorized: Literal[True] = True

    @field_validator("bindings")
    @classmethod
    def canonicalize_authorized_bindings(
        cls, bindings: tuple[StudyRunBinding, ...]
    ) -> tuple[StudyRunBinding, ...]:
        return _canonical_bindings(bindings)

    @model_validator(mode="after")
    def exact_authorization_types(self) -> StudySealedAuthorization:
        exact = (
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
        )
        for ref, expected, field_name in exact:
            if ref.media_type != expected:
                raise ValueError(f"{field_name} declares the wrong media type")
        _require_json_ref(self.analysis_plan_ref, field_name="analysis_plan_ref")
        return self


class StudyJournalEventContent(ImmutableModel):
    """Unsigned content of one linked study event."""

    schema_version: Literal["1"] = "1"
    study_controller_manifest_ref: ArtifactRef
    baseline_study_plan_ref: ArtifactRef
    plan_fingerprint: Sha256
    study_id: NonEmptyStr
    sequence: Annotated[int, Field(ge=0, strict=True)]
    previous_tail_ref: ArtifactRef | None
    kind: StudyEventKind
    from_state: StudyState | None
    to_state: StudyState
    run_key: StudyRunKey | None = None
    run_closure_ref: ArtifactRef | None = None
    barrier_closure_ref: ArtifactRef | None = None
    sealed_authorization_ref: ArtifactRef | None = None
    invalidation_ref: ArtifactRef | None = None
    reason: NonEmptyStr

    @model_validator(mode="after")
    def exact_event_shape(self) -> StudyJournalEventContent:
        if self.study_controller_manifest_ref.media_type != STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE:
            raise ValueError("study_controller_manifest_ref declares the wrong media type")
        if self.baseline_study_plan_ref.media_type != BASELINE_STUDY_PLAN_MEDIA_TYPE:
            raise ValueError("baseline_study_plan_ref declares the wrong media type")
        if self.sequence == 0 and self.previous_tail_ref is not None:
            raise ValueError("study sequence 0 must not have a previous tail")
        if self.sequence > 0 and self.previous_tail_ref is None:
            raise ValueError("study sequence greater than 0 requires a previous tail")
        if (
            self.previous_tail_ref is not None
            and self.previous_tail_ref.media_type != STUDY_JOURNAL_EVENT_MEDIA_TYPE
        ):
            raise ValueError("previous_tail_ref declares the wrong media type")

        if self.kind is StudyEventKind.INVALIDATED:
            if (
                self.from_state
                not in {
                    StudyState.FROZEN,
                    StudyState.COLLECTING,
                    StudyState.BARRIER_CLOSED,
                }
                or self.to_state is not StudyState.INVALIDATED
            ):
                raise ValueError("invalidated declares an illegal study transition")
        elif (self.from_state, self.to_state) != _EVENT_TRANSITIONS[self.kind]:
            raise ValueError(f"{self.kind.value} declares an illegal study transition")

        payloads = {
            "run_closure_ref": self.run_closure_ref,
            "barrier_closure_ref": self.barrier_closure_ref,
            "sealed_authorization_ref": self.sealed_authorization_ref,
            "invalidation_ref": self.invalidation_ref,
        }
        required = _EVENT_REQUIRED_REFS[self.kind]
        for field_name, value in payloads.items():
            if field_name in required and value is None:
                raise ValueError(f"{self.kind.value} requires {field_name}")
            if field_name not in required and value is not None:
                raise ValueError(f"{self.kind.value} must not include {field_name}")
        if (self.run_key is not None) != (self.kind is StudyEventKind.RUN_REGISTERED):
            raise ValueError("run_key is only required for run registration")
        exact_refs = (
            (self.run_closure_ref, STUDY_RUN_CLOSURE_MEDIA_TYPE, "run_closure_ref"),
            (
                self.barrier_closure_ref,
                STUDY_BARRIER_CLOSURE_MEDIA_TYPE,
                "barrier_closure_ref",
            ),
            (
                self.sealed_authorization_ref,
                STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE,
                "sealed_authorization_ref",
            ),
        )
        for ref, expected, field_name in exact_refs:
            if ref is not None and ref.media_type != expected:
                raise ValueError(f"{field_name} declares the wrong media type")
        if self.invalidation_ref is not None:
            media_type = self.invalidation_ref.media_type.partition(";")[0].strip().lower()
            if media_type != "application/json" and not media_type.endswith("+json"):
                raise ValueError("invalidation_ref must declare a JSON media type")
        return self


_EVENT_TRANSITIONS: dict[StudyEventKind, tuple[StudyState | None, StudyState]] = {
    StudyEventKind.FROZEN: (None, StudyState.FROZEN),
    StudyEventKind.COLLECTION_STARTED: (StudyState.FROZEN, StudyState.COLLECTING),
    StudyEventKind.RUN_REGISTERED: (StudyState.COLLECTING, StudyState.COLLECTING),
    StudyEventKind.BARRIER_CLOSED: (StudyState.COLLECTING, StudyState.BARRIER_CLOSED),
    StudyEventKind.SEALED_AUTHORIZED: (
        StudyState.BARRIER_CLOSED,
        StudyState.SEALED_AUTHORIZED,
    ),
    StudyEventKind.INVALIDATED: (StudyState.COLLECTING, StudyState.INVALIDATED),
}
_EVENT_REQUIRED_REFS: dict[StudyEventKind, frozenset[str]] = {
    kind: frozenset() for kind in StudyEventKind
}
_EVENT_REQUIRED_REFS.update(
    {
        StudyEventKind.RUN_REGISTERED: frozenset({"run_closure_ref"}),
        StudyEventKind.BARRIER_CLOSED: frozenset({"barrier_closure_ref"}),
        StudyEventKind.SEALED_AUTHORIZED: frozenset({"sealed_authorization_ref"}),
        StudyEventKind.INVALIDATED: frozenset({"invalidation_ref"}),
    }
)


class StudyJournalEvent(StudyJournalEventContent):
    """One HMAC-attested study event and linked journal tail."""

    attestor_id: Sha256
    attestation_sha256: Sha256

    @property
    def content(self) -> StudyJournalEventContent:
        return StudyJournalEventContent.model_validate(
            self.model_dump(
                mode="python",
                exclude={"attestor_id", "attestation_sha256"},
                round_trip=True,
                warnings="none",
            ),
            strict=True,
        )


def _attestor_id(secret: bytes) -> str:
    return hashlib.sha256(_EVENT_ATTESTOR_ID_DOMAIN + secret).hexdigest()


def _attestation(secret: bytes, content: StudyJournalEventContent, attestor_id: str) -> str:
    return hmac.new(
        secret,
        _EVENT_ATTESTATION_DOMAIN
        + attestor_id.encode("ascii")
        + b"\x00"
        + canonical_json_bytes(content),
        hashlib.sha256,
    ).hexdigest()


class StudyEventVerificationCapability:
    """Verify study events from one trusted controller signer."""

    __slots__ = ("__secret", "_attestor_id")

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("study event verification secret must contain at least 32 bytes")
        self.__secret = secret
        self._attestor_id = _attestor_id(secret)

    @property
    def attestor_id(self) -> str:
        return self._attestor_id

    def verify(self, event: StudyJournalEvent | object) -> StudyJournalEvent:
        validated = StudyJournalEvent.model_validate(event)
        if validated.attestor_id != self._attestor_id:
            raise StudyEventAttestationError("study event was signed by a foreign attestor")
        expected = _attestation(self.__secret, validated.content, self._attestor_id)
        if not hmac.compare_digest(validated.attestation_sha256, expected):
            raise StudyEventAttestationError("study event HMAC attestation is invalid")
        return validated


class TrustedStudyEventService:
    """Process-scoped signing capability retained by trusted study control."""

    __slots__ = ("__secret", "_verification_capability")

    def __init__(self, secret: bytes | None = None) -> None:
        material = secrets.token_bytes(32) if secret is None else secret
        if not isinstance(material, bytes) or len(material) < 32:
            raise ValueError("study event signing secret must contain at least 32 bytes")
        self.__secret = material
        self._verification_capability = StudyEventVerificationCapability(material)

    @property
    def attestor_id(self) -> str:
        return self._verification_capability.attestor_id

    @property
    def verification_capability(self) -> StudyEventVerificationCapability:
        return self._verification_capability

    def _issue(self, content: StudyJournalEventContent) -> StudyJournalEvent:
        validated = StudyJournalEventContent.model_validate(content)
        return StudyJournalEvent(
            **validated.model_dump(mode="python", round_trip=True, warnings="none"),
            attestor_id=self.attestor_id,
            attestation_sha256=_attestation(self.__secret, validated, self.attestor_id),
        )


@dataclass(frozen=True, slots=True)
class _LiveRunCapabilities:
    """Process-local capabilities retained for final pre-sealed revalidation."""

    search_controller: SearchController
    search_selection_tail_ref: ArtifactRef
    experiment_controller: ExperimentController
    experiment_selection_tail_ref: ArtifactRef
    automatic_search_loop: object
    automatic_search_result_ref: ArtifactRef


class StudySnapshot(ImmutableModel):
    """Replay-derived state at one exact study journal tail."""

    study_controller_manifest_ref: ArtifactRef
    baseline_study_plan_ref: ArtifactRef
    plan_fingerprint: Sha256
    study_id: NonEmptyStr
    tail_ref: ArtifactRef
    sequence: Annotated[int, Field(ge=0, strict=True)]
    event_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    state: StudyState
    expected_keys: Annotated[tuple[StudyRunKey, ...], Field(min_length=1)]
    bindings: tuple[StudyRunBinding, ...]
    missing_keys: tuple[StudyRunKey, ...]
    barrier_closure_ref: ArtifactRef | None
    sealed_authorization_ref: ArtifactRef | None
    run_registration_open: bool
    proposal_api_open: bool


class StudyJournal:
    """Replay structural and semantic integrity of one attested study chain."""

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        study_controller_manifest_ref: ArtifactRef,
        event_verifier: StudyEventVerificationCapability,
    ) -> None:
        self.store = store
        self.study_controller_manifest_ref = ArtifactRef.model_validate(
            study_controller_manifest_ref
        )
        if self.study_controller_manifest_ref.media_type != STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE:
            raise StudyJournalIntegrityError(
                "study controller manifest ref declares the wrong media type"
            )
        self.manifest = self.store.get_json(
            self.study_controller_manifest_ref, StudyControllerManifest
        )
        if type(event_verifier) is not StudyEventVerificationCapability:
            raise TypeError("event_verifier must be an exact StudyEventVerificationCapability")
        if event_verifier.attestor_id != self.manifest.journal_attestor_id:
            raise StudyEventAttestationError(
                "study event verifier does not match the frozen attestor"
            )
        self.event_verifier = event_verifier
        self.plan = self.store.get_json(self.manifest.baseline_study_plan_ref, BaselineStudyPlan)
        self.expected_keys = expected_run_keys(self.plan)
        self.expected_run_manifests = self._verify_frozen_run_schedule()

    def _verify_frozen_run_schedule(
        self,
    ) -> dict[tuple[str, int], tuple[StudyExpectedRunManifest, SearchRunManifest]]:
        """Strict-load the complete schedule before the study journal can open."""

        manifest_keys = tuple(run.key for run in self.manifest.expected_run_manifests)
        if manifest_keys != self.expected_keys:
            raise StudyJournalIntegrityError(
                "expected_run_manifests must cover exactly the plan-derived run keys"
            )
        try:
            self.store.get_json(self.manifest.analysis_plan_ref)
        except Exception as exc:
            raise StudyJournalIntegrityError(
                f"frozen common analysis plan could not be verified: {exc}"
            ) from exc

        verified: dict[tuple[str, int], tuple[StudyExpectedRunManifest, SearchRunManifest]] = {}
        policies: dict[tuple[str, int], SearchPolicy] = {}
        experiments: dict[tuple[str, int], ExperimentManifest] = {}

        for binding in self.manifest.expected_run_manifests:
            key_tuple = _key_order(binding.key)
            try:
                run = self.store.get_json(binding.search_run_ref, SearchRunManifest)
                policy = self.store.get_json(run.search_policy_ref, SearchPolicy)
                stopping = self.store.get_json(run.stopping_policy_ref, SearchStoppingPolicy)
                plugin = self.store.get_json(run.strategy_plugin_ref, StrategyPluginManifest)
                experiment = self.store.get_json(run.experiment_ref, ExperimentManifest)
                self.store.get_bytes(plugin.implementation_ref)
                if run.prompt_mutation_catalogue_ref is not None:
                    self.store.get_json(run.prompt_mutation_catalogue_ref)
            except Exception as exc:
                raise StudyJournalIntegrityError(
                    "pre-registered run artifacts could not be strictly loaded for "
                    f"{binding.key.baseline_kind.value}/seed={binding.key.search_run_seed}: {exc}"
                ) from exc

            arm = self.plan.arm(binding.key.baseline_kind)
            expected_coordinates = (
                self.manifest.baseline_study_plan_ref,
                binding.key.baseline_kind,
                binding.key.search_run_seed,
                arm.evaluation.repeat_seeds,
                arm.context.proposal_random_seed,
                arm.context.seed_harness_ref,
                self.plan.fingerprint,
                self.manifest.analysis_plan_ref,
            )
            actual_coordinates = (
                run.baseline_study_plan_ref,
                run.baseline_kind,
                run.search_run_seed,
                run.repeat_seeds,
                run.proposal_master_seed,
                run.seed_harness_ref,
                run.baseline_plan_fingerprint,
                run.analysis_plan_ref,
            )
            if actual_coordinates != expected_coordinates:
                raise StudyJournalIntegrityError(
                    "pre-registered search run drifted from its exact study cell"
                )
            if policy.baseline_kind is not binding.key.baseline_kind:
                raise StudyJournalIntegrityError(
                    "pre-registered search policy belongs to another baseline"
                )
            if stopping != policy.stopping_policy:
                raise StudyJournalIntegrityError(
                    "pre-registered stopping policy differs from the search policy"
                )
            if (
                plugin.baseline_kind is not binding.key.baseline_kind
                or plugin.consumes_feedback != policy.available_feedback
                or plugin.mutation != policy.mutation
            ):
                raise StudyJournalIntegrityError(
                    "pre-registered strategy plugin differs from the search policy"
                )
            if experiment.seed_harness_ref != run.seed_harness_ref:
                raise StudyJournalIntegrityError(
                    "pre-registered experiment and search run use different seed harnesses"
                )
            required_baselines = frozenset(kind.value for kind in BaselineKind)
            if frozenset(experiment.baselines) != required_baselines:
                raise StudyJournalIntegrityError(
                    "pre-registered experiment does not freeze the exact four baselines"
                )
            verified[key_tuple] = (binding, run)
            policies[key_tuple] = policy
            experiments[key_tuple] = experiment

        self._verify_cross_run_profiles(
            verified=verified,
            policies=policies,
            experiments=experiments,
        )
        return verified

    @staticmethod
    def _verify_cross_run_profiles(
        *,
        verified: dict[tuple[str, int], tuple[StudyExpectedRunManifest, SearchRunManifest]],
        policies: dict[tuple[str, int], SearchPolicy],
        experiments: dict[tuple[str, int], ExperimentManifest],
    ) -> None:
        """Reject seed-level treatment drift and unmatched search resources."""

        for kind in BaselineKind:
            keys = tuple(key for key in verified if key[0] == kind.value)
            first = keys[0]
            first_run = verified[first][1]
            for key in keys[1:]:
                run = verified[key][1]
                if (
                    run.search_policy_ref != first_run.search_policy_ref
                    or run.stopping_policy_ref != first_run.stopping_policy_ref
                    or run.strategy_plugin_ref != first_run.strategy_plugin_ref
                    or run.prompt_mutation_catalogue_ref != first_run.prompt_mutation_catalogue_ref
                ):
                    raise StudyJournalIntegrityError(
                        f"{kind.value} seeds must share one frozen policy/plugin profile"
                    )

        non_static = tuple(key for key in verified if key[0] != BaselineKind.STATIC.value)
        anchor_policy = policies[non_static[0]]
        excluded = {"baseline_kind", "available_feedback", "mutation", "max_diagnoses"}
        anchor_profile = anchor_policy.model_dump(
            mode="python", exclude=excluded, round_trip=True, warnings="none"
        )
        for key in non_static[1:]:
            profile = policies[key].model_dump(
                mode="python", exclude=excluded, round_trip=True, warnings="none"
            )
            if profile != anchor_profile:
                raise StudyJournalIntegrityError(
                    "non-static baselines must share comparable search/resource limits"
                )

        experiment_anchor = next(iter(experiments.values()))
        for experiment in tuple(experiments.values())[1:]:
            if experiment != experiment_anchor:
                raise StudyJournalIntegrityError(
                    "pre-registered experiment execution profiles drifted across study cells"
                )

    def replay(self, tail_ref: ArtifactRef) -> StudySnapshot:
        validated_tail = ArtifactRef.model_validate(tail_ref)
        backwards: list[tuple[ArtifactRef, StudyJournalEvent]] = []
        seen: set[str] = set()
        cursor: ArtifactRef | None = validated_tail
        while cursor is not None:
            if cursor.media_type != STUDY_JOURNAL_EVENT_MEDIA_TYPE:
                raise StudyJournalIntegrityError("study tail declares the wrong media type")
            if cursor.sha256 in seen:
                raise StudyJournalCycleError(f"study journal cycle detected at {cursor.sha256}")
            seen.add(cursor.sha256)
            event = StudyEventVerificationCapability.verify(
                self.event_verifier,
                self.store.get_json(cursor, StudyJournalEvent),
            )
            backwards.append((cursor, event))
            cursor = event.previous_tail_ref

        events = tuple(reversed(backwards))
        for expected_sequence, (_, event) in enumerate(events):
            self._verify_event_coordinates(event)
            if event.sequence != expected_sequence:
                raise StudyJournalIntegrityError(
                    "study sequence is not contiguous from zero: "
                    f"expected {expected_sequence}, got {event.sequence}"
                )
            expected_previous = None if expected_sequence == 0 else events[expected_sequence - 1][0]
            if event.previous_tail_ref != expected_previous:
                raise StudyJournalIntegrityError(
                    f"study link is inconsistent at sequence {event.sequence}"
                )
        return self._replay_semantics(events)

    def _verify_event_coordinates(self, event: StudyJournalEvent) -> None:
        expected = (
            self.study_controller_manifest_ref,
            self.manifest.baseline_study_plan_ref,
            self.plan.fingerprint,
            self.manifest.study_id,
            self.manifest.journal_attestor_id,
        )
        actual = (
            event.study_controller_manifest_ref,
            event.baseline_study_plan_ref,
            event.plan_fingerprint,
            event.study_id,
            event.attestor_id,
        )
        if actual != expected:
            raise StudyJournalIntegrityError(
                f"foreign study coordinates at sequence {event.sequence}"
            )

    def _verify_run_closure(self, ref: ArtifactRef, key: StudyRunKey) -> StudyRunClosure:
        if ref.media_type != STUDY_RUN_CLOSURE_MEDIA_TYPE:
            raise StudyJournalIntegrityError("run closure ref declares the wrong media type")
        closure = self.store.get_json(ref, StudyRunClosure)
        expected_run_binding, expected_run = self.expected_run_manifests[_key_order(key)]
        expected = (
            self.study_controller_manifest_ref,
            self.manifest.baseline_study_plan_ref,
            self.plan.fingerprint,
            key,
            expected_run_binding.search_run_ref,
            expected_run.experiment_ref,
            self.manifest.analysis_plan_ref,
        )
        actual = (
            closure.study_controller_manifest_ref,
            closure.baseline_study_plan_ref,
            closure.plan_fingerprint,
            closure.key,
            closure.search_run_ref,
            closure.experiment_ref,
            closure.analysis_plan_ref,
        )
        if actual != expected:
            raise StudyJournalIntegrityError("run closure belongs to a foreign study or key")
        try:
            SearchRunAdmissionService(self.store).verify_report(
                report_ref=closure.search_run_admission_report_ref,
                search_run_ref=closure.search_run_ref,
                controller_manifest_ref=closure.search_controller_manifest_ref,
                expectation=SearchRunAdmissionExpectation(
                    baseline_study_plan_ref=self.manifest.baseline_study_plan_ref,
                    experiment_ref=closure.experiment_ref,
                    baseline_kind=key.baseline_kind,
                    search_run_seed=key.search_run_seed,
                ),
            )
            result_payload = self.store.get_bytes(closure.automatic_search_result_ref)
            automatic_result = self.store.get_json(
                closure.automatic_search_result_ref,
                AutomaticSearchLoopResult,
            )
            if result_payload != canonical_json_bytes(automatic_result):
                raise ValueError("automatic search result is not canonical")
        except Exception as exc:
            raise StudyJournalIntegrityError(
                f"run automatic-search execution proof could not be replayed: {exc}"
            ) from exc
        search_closure = self.store.get_json(
            closure.search_selection_closure_ref, SearchSelectionClosure
        )
        search_manifest = self.store.get_json(
            closure.search_controller_manifest_ref, SearchControllerManifest
        )
        if (
            search_manifest.search_run_ref != expected_run_binding.search_run_ref
            or search_manifest.experiment_ref != expected_run.experiment_ref
            or search_manifest.analysis_plan_ref != self.manifest.analysis_plan_ref
        ):
            raise StudyJournalIntegrityError(
                "run search controller differs from the pre-registered manifest"
            )
        experiment_events = ExperimentJournal(self.store).replay(
            closure.experiment_selection_tail_ref
        )
        if experiment_events[-1].to_state is not ExperimentState.SELECTION_CLOSED:
            raise StudyJournalIntegrityError("run experiment tail is not selection closed")
        if experiment_events[-1].experiment_ref != closure.experiment_ref:
            raise StudyJournalIntegrityError("run experiment tail belongs to another experiment")
        evidence_refs = experiment_events[-1].evidence_refs
        if evidence_refs != (closure.experiment_selection_closure_ref,):
            raise StudyJournalIntegrityError(
                "run experiment tail does not bind its selection closure"
            )
        experiment_closure = self.store.get_json(
            closure.experiment_selection_closure_ref, SelectionClosure
        )
        if (
            search_closure.champion_harness_ref != closure.champion_harness_ref
            or search_closure.champion_candidate_ref != closure.champion_candidate_ref
            or search_closure.analysis_plan_ref != closure.analysis_plan_ref
            or experiment_closure.champion_harness_ref != closure.champion_harness_ref
            or experiment_closure.champion_candidate_ref != closure.champion_candidate_ref
            or experiment_closure.analysis_plan_ref != closure.analysis_plan_ref
            or experiment_closure.usage_tail_ref != closure.usage_tail_ref
            or experiment_closure.experiment_ref != closure.experiment_ref
        ):
            raise StudyJournalIntegrityError("run closure selection joins are inconsistent")
        automatic_coordinates = (
            automatic_result.search_run_ref,
            automatic_result.admission_report_ref,
            automatic_result.final_search_tail_ref,
            automatic_result.search_selection_closure_ref,
            automatic_result.final_experiment_tail_ref,
            automatic_result.final_champion_harness_ref,
            automatic_result.final_champion_candidate_ref,
            automatic_result.completed_rounds,
            automatic_result.gate_nominations,
        )
        expected_automatic_coordinates = (
            closure.search_run_ref,
            closure.search_run_admission_report_ref,
            closure.search_selection_tail_ref,
            closure.search_selection_closure_ref,
            closure.experiment_selection_tail_ref,
            closure.champion_harness_ref,
            closure.champion_candidate_ref,
            search_closure.completed_rounds,
            search_closure.gate_nominations,
        )
        if automatic_coordinates != expected_automatic_coordinates:
            raise StudyJournalIntegrityError(
                "run automatic-search result differs from its controller closures"
            )
        if automatic_result.admission_report_ref not in automatic_result.archived_artifact_refs:
            raise StudyJournalIntegrityError(
                "run automatic-search result omitted its admission proof"
            )
        return closure

    def _verify_barrier_closure(
        self,
        ref: ArtifactRef,
        *,
        collection_tail_ref: ArtifactRef,
        bindings: tuple[StudyRunBinding, ...],
    ) -> StudyBarrierClosure:
        if ref.media_type != STUDY_BARRIER_CLOSURE_MEDIA_TYPE:
            raise StudyJournalIntegrityError("barrier closure ref declares the wrong media type")
        closure = self.store.get_json(ref, StudyBarrierClosure)
        expected = (
            self.study_controller_manifest_ref,
            self.manifest.baseline_study_plan_ref,
            self.plan.fingerprint,
            self.manifest.analysis_plan_ref,
            collection_tail_ref,
            self.expected_keys,
            bindings,
        )
        actual = (
            closure.study_controller_manifest_ref,
            closure.baseline_study_plan_ref,
            closure.plan_fingerprint,
            closure.analysis_plan_ref,
            closure.collection_tail_ref,
            closure.expected_keys,
            closure.bindings,
        )
        if actual != expected:
            raise StudyJournalIntegrityError("study barrier closure is inconsistent")
        return closure

    def _verify_sealed_authorization(
        self,
        ref: ArtifactRef,
        *,
        barrier_tail_ref: ArtifactRef,
        barrier_closure_ref: ArtifactRef,
        bindings: tuple[StudyRunBinding, ...],
    ) -> StudySealedAuthorization:
        if ref.media_type != STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE:
            raise StudyJournalIntegrityError(
                "study sealed authorization ref declares the wrong media type"
            )
        authorization = self.store.get_json(ref, StudySealedAuthorization)
        expected = (
            self.study_controller_manifest_ref,
            self.manifest.baseline_study_plan_ref,
            self.plan.fingerprint,
            self.manifest.analysis_plan_ref,
            barrier_tail_ref,
            barrier_closure_ref,
            bindings,
        )
        actual = (
            authorization.study_controller_manifest_ref,
            authorization.baseline_study_plan_ref,
            authorization.plan_fingerprint,
            authorization.analysis_plan_ref,
            authorization.barrier_closed_tail_ref,
            authorization.barrier_closure_ref,
            authorization.bindings,
        )
        if actual != expected:
            raise StudyJournalIntegrityError("study sealed authorization is inconsistent")
        return authorization

    def _replay_semantics(
        self, events: tuple[tuple[ArtifactRef, StudyJournalEvent], ...]
    ) -> StudySnapshot:
        if not events:
            raise StudyJournalIntegrityError("study journal is empty")
        state: StudyState | None = None
        by_key: dict[tuple[str, int], StudyRunBinding] = {}
        barrier_closure_ref: ArtifactRef | None = None
        sealed_authorization_ref: ArtifactRef | None = None

        for index, (_, event) in enumerate(events):
            if event.from_state is not state:
                raise StudyJournalIntegrityError(
                    f"study state discontinuity at sequence {event.sequence}"
                )
            if event.kind is StudyEventKind.FROZEN:
                if index != 0:
                    raise StudyJournalIntegrityError("FROZEN must be the study root")
            elif event.kind is StudyEventKind.RUN_REGISTERED:
                assert event.run_key is not None and event.run_closure_ref is not None
                if event.run_key not in self.expected_keys:
                    raise StudyJournalIntegrityError("registered run key is not in the study plan")
                key_tuple = _key_order(event.run_key)
                if key_tuple in by_key:
                    raise StudyJournalIntegrityError("study run key was registered more than once")
                closure = self._verify_run_closure(event.run_closure_ref, event.run_key)
                by_key[key_tuple] = StudyRunBinding(
                    key=event.run_key,
                    run_closure_ref=event.run_closure_ref,
                )
                if closure.key != event.run_key:  # pragma: no cover - helper enforces
                    raise StudyJournalIntegrityError("run closure key changed")
            elif event.kind is StudyEventKind.BARRIER_CLOSED:
                bindings = _canonical_bindings(tuple(by_key.values()))
                if tuple(binding.key for binding in bindings) != self.expected_keys:
                    raise StudyJournalIntegrityError("study barrier closed with missing runs")
                assert event.barrier_closure_ref is not None
                assert event.previous_tail_ref is not None
                self._verify_barrier_closure(
                    event.barrier_closure_ref,
                    collection_tail_ref=event.previous_tail_ref,
                    bindings=bindings,
                )
                barrier_closure_ref = event.barrier_closure_ref
            elif event.kind is StudyEventKind.SEALED_AUTHORIZED:
                if barrier_closure_ref is None or event.previous_tail_ref is None:
                    raise StudyJournalIntegrityError("sealed authorization has no closed barrier")
                bindings = _canonical_bindings(tuple(by_key.values()))
                assert event.sealed_authorization_ref is not None
                self._verify_sealed_authorization(
                    event.sealed_authorization_ref,
                    barrier_tail_ref=event.previous_tail_ref,
                    barrier_closure_ref=barrier_closure_ref,
                    bindings=bindings,
                )
                sealed_authorization_ref = event.sealed_authorization_ref
            elif event.kind is StudyEventKind.INVALIDATED:
                assert event.invalidation_ref is not None
                self.store.get_json(event.invalidation_ref)
            state = event.to_state

        assert state is not None
        bindings = _canonical_bindings(tuple(by_key.values())) if by_key else ()
        supplied = frozenset(binding.key for binding in bindings)
        missing = tuple(key for key in self.expected_keys if key not in supplied)
        registrations_open = state is StudyState.COLLECTING and bool(missing)
        return StudySnapshot(
            study_controller_manifest_ref=self.study_controller_manifest_ref,
            baseline_study_plan_ref=self.manifest.baseline_study_plan_ref,
            plan_fingerprint=self.plan.fingerprint,
            study_id=self.manifest.study_id,
            tail_ref=events[-1][0],
            sequence=events[-1][1].sequence,
            event_refs=tuple(ref for ref, _ in events),
            state=state,
            expected_keys=self.expected_keys,
            bindings=bindings,
            missing_keys=missing,
            barrier_closure_ref=barrier_closure_ref,
            sealed_authorization_ref=sealed_authorization_ref,
            run_registration_open=registrations_open,
            proposal_api_open=registrations_open,
        )


class StudyController:
    """Single-writer study barrier over every plan-derived search run."""

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        study_controller_manifest_ref: ArtifactRef,
        event_service: TrustedStudyEventService,
        tail_ref: ArtifactRef | None = None,
    ) -> None:
        if type(event_service) is not TrustedStudyEventService:
            raise TypeError("event_service must be an exact TrustedStudyEventService")
        self.store = store
        self.study_controller_manifest_ref = ArtifactRef.model_validate(
            study_controller_manifest_ref
        )
        self.event_service = event_service
        self.journal = StudyJournal(
            store,
            study_controller_manifest_ref=self.study_controller_manifest_ref,
            event_verifier=event_service.verification_capability,
        )
        self.manifest = self.journal.manifest
        self.plan = self.journal.plan
        self.expected_keys = self.journal.expected_keys
        self._tail_ref: ArtifactRef | None = None
        self._snapshot: StudySnapshot | None = None
        self._live_runs: dict[tuple[str, int], _LiveRunCapabilities] = {}
        if tail_ref is not None:
            self._snapshot = self.journal.replay(ArtifactRef.model_validate(tail_ref))
            self._tail_ref = self._snapshot.tail_ref

    @property
    def tail_ref(self) -> ArtifactRef | None:
        return self._tail_ref

    @property
    def snapshot(self) -> StudySnapshot | None:
        return self._snapshot

    @property
    def state(self) -> StudyState | None:
        return None if self._snapshot is None else self._snapshot.state

    def freeze(self) -> ArtifactRef:
        if self._tail_ref is not None:
            raise StaleStudyTailError("study controller is already frozen")
        return self._append(
            kind=StudyEventKind.FROZEN,
            previous_tail_ref=None,
            reason="baseline plan and complete run-key schedule frozen",
        )

    def start_collection(self, *, previous_tail_ref: ArtifactRef) -> ArtifactRef:
        snapshot = self._require_tail(previous_tail_ref)
        self._require_state(snapshot, StudyState.FROZEN)
        return self._append(
            kind=StudyEventKind.COLLECTION_STARTED,
            previous_tail_ref=previous_tail_ref,
            reason="study run-closure collection opened",
        )

    def assert_proposal_api_open(self, *, previous_tail_ref: ArtifactRef) -> StudySnapshot:
        snapshot = self._require_tail(previous_tail_ref)
        if not snapshot.proposal_api_open:
            raise StudyBarrierError("study proposal API is permanently closed")
        return snapshot

    def register_run_closure(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        automatic_search_loop: object,
        automatic_search_result_ref: ArtifactRef,
        search_controller: SearchController,
        search_selection_tail_ref: ArtifactRef,
        experiment_controller: ExperimentController,
        experiment_selection_tail_ref: ArtifactRef,
    ) -> ArtifactRef:
        snapshot = self._require_tail(previous_tail_ref)
        self._require_state(snapshot, StudyState.COLLECTING)
        if not snapshot.run_registration_open:
            raise StudyBarrierError("study run registration is permanently closed")
        if type(search_controller) is not SearchController:
            raise TypeError("search_controller must be an exact SearchController")
        if type(experiment_controller) is not ExperimentController:
            raise TypeError("experiment_controller must be an exact ExperimentController")
        if type(automatic_search_loop) is not AutomaticSearchLoop:
            raise TypeError("automatic_search_loop must be an exact AutomaticSearchLoop")
        automatic_result_ref = ArtifactRef.model_validate(automatic_search_result_ref)
        search_tail = ArtifactRef.model_validate(search_selection_tail_ref)
        experiment_tail = ArtifactRef.model_validate(experiment_selection_tail_ref)
        try:
            search_snapshot = SearchController.verify_selection_closure(
                search_controller, search_tail
            )
        except Exception as exc:
            raise StudyBarrierError(f"search selection verification failed: {exc}") from exc
        try:
            experiment_closure = ExperimentController.verify_experiment_selection_closure(
                experiment_controller, experiment_tail
            )
        except Exception as exc:
            raise StudyBarrierError(f"experiment selection verification failed: {exc}") from exc
        if search_snapshot.state is not SearchState.SELECTION_CLOSED:
            raise StudyBarrierError("search run is not selection closed")
        key = StudyRunKey(
            baseline_kind=search_snapshot.baseline_kind,
            search_run_seed=search_snapshot.search_seed,
        )
        if key not in self.expected_keys:
            raise StudyBarrierError("search run key is foreign to the baseline study plan")
        if key in frozenset(binding.key for binding in snapshot.bindings):
            raise StudyBarrierError("study run key is already registered")
        expected_binding, expected_run = self.journal.expected_run_manifests[_key_order(key)]
        if search_snapshot.search_run_ref != expected_binding.search_run_ref:
            raise StudyBarrierError(
                "search controller does not bind the pre-registered search-run manifest"
            )

        search_manifest = self.store.get_json(
            search_snapshot.controller_manifest_ref, SearchControllerManifest
        )
        if search_controller.controller_manifest_ref != search_snapshot.controller_manifest_ref:
            raise StudyBarrierError("search verifier returned another controller manifest")
        if search_manifest.study_ref != self.manifest.baseline_study_plan_ref:
            raise StudyBarrierError("search controller belongs to another study")
        if (
            search_manifest.search_run_ref != expected_binding.search_run_ref
            or search_manifest.experiment_ref != expected_run.experiment_ref
            or search_manifest.analysis_plan_ref != self.manifest.analysis_plan_ref
        ):
            raise StudyBarrierError(
                "search controller differs from the pre-registered run manifest"
            )
        if (
            search_manifest.experiment_ref != experiment_closure.experiment_ref
            or experiment_controller.experiment_ref != expected_run.experiment_ref
        ):
            raise StudyBarrierError("search and experiment controllers are not paired")
        if search_snapshot.selection_closure_ref is None:
            raise StudyBarrierError("search selection closure artifact is missing")
        search_closure = self.store.get_json(
            search_snapshot.selection_closure_ref, SearchSelectionClosure
        )
        search_run = self.store.get_json(search_manifest.search_run_ref, SearchRunManifest)
        arm = self.plan.arm(key.baseline_kind)
        expected_run_fields = (
            self.manifest.baseline_study_plan_ref,
            key.baseline_kind,
            key.search_run_seed,
            arm.evaluation.repeat_seeds,
            arm.context.proposal_random_seed,
            arm.context.seed_harness_ref,
            self.plan.fingerprint,
            self.manifest.analysis_plan_ref,
            expected_run.experiment_ref,
        )
        actual_run_fields = (
            search_run.baseline_study_plan_ref,
            search_run.baseline_kind,
            search_run.search_run_seed,
            search_run.repeat_seeds,
            search_run.proposal_master_seed,
            search_run.seed_harness_ref,
            search_run.baseline_plan_fingerprint,
            search_run.analysis_plan_ref,
            search_run.experiment_ref,
        )
        if actual_run_fields != expected_run_fields:
            raise StudyBarrierError("search run manifest drifted from the baseline study plan")

        experiment_events = ExperimentJournal(self.store).replay(experiment_tail)
        if experiment_events[-1].to_state is not ExperimentState.SELECTION_CLOSED:
            raise StudyBarrierError("experiment verifier returned a non-closed tail")
        if experiment_events[-1].experiment_ref != experiment_closure.experiment_ref:
            raise StudyBarrierError("experiment closure belongs to another tail")
        if len(experiment_events[-1].evidence_refs) != 1:
            raise StudyBarrierError("experiment selection event has invalid evidence shape")
        experiment_closure_ref = experiment_events[-1].evidence_refs[0]
        if experiment_closure_ref.media_type != SELECTION_CLOSURE_MEDIA_TYPE:
            raise StudyBarrierError("experiment selection event binds the wrong closure type")
        stored_experiment_closure = self.store.get_json(experiment_closure_ref, SelectionClosure)
        if stored_experiment_closure != experiment_closure:
            raise StudyBarrierError(
                "experiment verifier result does not match the referenced closure"
            )
        expected_selection_fields = (
            expected_run.experiment_ref,
            self.manifest.analysis_plan_ref,
            search_snapshot.champion_harness_ref,
            search_snapshot.champion_candidate_ref,
        )
        search_fields = (
            search_closure.experiment_ref,
            search_closure.analysis_plan_ref,
            search_closure.champion_harness_ref,
            search_closure.champion_candidate_ref,
        )
        experiment_fields = (
            experiment_closure.experiment_ref,
            experiment_closure.analysis_plan_ref,
            experiment_closure.champion_harness_ref,
            experiment_closure.champion_candidate_ref,
        )
        if (
            search_fields != expected_selection_fields
            or experiment_fields != expected_selection_fields
        ):
            raise StudyBarrierError(
                "search and experiment selection closures do not select the same lineage"
            )
        try:
            automatic_result = AutomaticSearchLoop.verify_execution(
                automatic_search_loop,
                automatic_result_ref,
                search_run_ref=expected_binding.search_run_ref,
                controller=search_controller,
                search_selection_tail_ref=search_tail,
                lifecycle=experiment_controller,
                experiment_selection_tail_ref=experiment_tail,
            )
        except Exception as exc:
            raise StudyBarrierError(
                f"automatic search execution verification failed: {exc}"
            ) from exc

        for binding in snapshot.bindings:
            registered = self.store.get_json(binding.run_closure_ref, StudyRunClosure)
            if (
                registered.search_controller_manifest_ref == search_snapshot.controller_manifest_ref
                or registered.search_selection_tail_ref == search_tail
            ):
                raise StudyBarrierError(
                    "independent study cells must not reuse controller/run capabilities"
                )
        if any(
            live.search_controller is search_controller
            or live.experiment_controller is experiment_controller
            or live.automatic_search_loop is automatic_search_loop
            for live in self._live_runs.values()
        ):
            raise StudyBarrierError(
                "independent study cells must use distinct live controller instances"
            )

        run_closure = StudyRunClosure(
            study_controller_manifest_ref=self.study_controller_manifest_ref,
            baseline_study_plan_ref=self.manifest.baseline_study_plan_ref,
            plan_fingerprint=self.plan.fingerprint,
            key=key,
            search_controller_manifest_ref=search_snapshot.controller_manifest_ref,
            search_run_ref=search_snapshot.search_run_ref,
            search_run_admission_report_ref=automatic_result.admission_report_ref,
            automatic_search_result_ref=automatic_result_ref,
            search_selection_tail_ref=search_tail,
            search_selection_closure_ref=search_snapshot.selection_closure_ref,
            experiment_ref=experiment_closure.experiment_ref,
            experiment_selection_tail_ref=experiment_tail,
            experiment_selection_closure_ref=experiment_closure_ref,
            champion_harness_ref=experiment_closure.champion_harness_ref,
            champion_candidate_ref=experiment_closure.champion_candidate_ref,
            analysis_plan_ref=experiment_closure.analysis_plan_ref,
            usage_tail_ref=experiment_closure.usage_tail_ref,
        )
        run_closure_ref = self.store.put_json(run_closure, media_type=STUDY_RUN_CLOSURE_MEDIA_TYPE)
        next_tail = self._append(
            kind=StudyEventKind.RUN_REGISTERED,
            previous_tail_ref=previous_tail_ref,
            run_key=key,
            run_closure_ref=run_closure_ref,
            reason=f"closed run registered: {key.baseline_kind.value}/seed={key.search_run_seed}",
        )
        self._live_runs[_key_order(key)] = _LiveRunCapabilities(
            search_controller=search_controller,
            search_selection_tail_ref=search_tail,
            experiment_controller=experiment_controller,
            experiment_selection_tail_ref=experiment_tail,
            automatic_search_loop=automatic_search_loop,
            automatic_search_result_ref=automatic_result_ref,
        )
        return next_tail

    def close_selection_barrier(self, *, previous_tail_ref: ArtifactRef) -> ArtifactRef:
        snapshot = self._require_tail(previous_tail_ref)
        self._require_state(snapshot, StudyState.COLLECTING)
        if snapshot.missing_keys:
            missing = ", ".join(
                f"{key.baseline_kind.value}/seed={key.search_run_seed}"
                for key in snapshot.missing_keys
            )
            raise StudyBarrierError(f"study selection barrier has missing runs: {missing}")
        closure = StudyBarrierClosure(
            study_controller_manifest_ref=self.study_controller_manifest_ref,
            baseline_study_plan_ref=self.manifest.baseline_study_plan_ref,
            plan_fingerprint=self.plan.fingerprint,
            analysis_plan_ref=self.manifest.analysis_plan_ref,
            collection_tail_ref=snapshot.tail_ref,
            expected_keys=self.expected_keys,
            bindings=snapshot.bindings,
        )
        closure_ref = self.store.put_json(closure, media_type=STUDY_BARRIER_CLOSURE_MEDIA_TYPE)
        return self._append(
            kind=StudyEventKind.BARRIER_CLOSED,
            previous_tail_ref=previous_tail_ref,
            barrier_closure_ref=closure_ref,
            reason="all four-arm search-run selections closed",
        )

    def authorize_sealed(self, *, previous_tail_ref: ArtifactRef) -> ArtifactRef:
        snapshot = self._require_tail(previous_tail_ref)
        self._require_state(snapshot, StudyState.BARRIER_CLOSED)
        if snapshot.barrier_closure_ref is None:
            raise StudyJournalIntegrityError("study barrier closure artifact is missing")
        self._reverify_all_live_runs(snapshot)
        authorization = StudySealedAuthorization(
            study_controller_manifest_ref=self.study_controller_manifest_ref,
            baseline_study_plan_ref=self.manifest.baseline_study_plan_ref,
            plan_fingerprint=self.plan.fingerprint,
            analysis_plan_ref=self.manifest.analysis_plan_ref,
            barrier_closed_tail_ref=snapshot.tail_ref,
            barrier_closure_ref=snapshot.barrier_closure_ref,
            bindings=snapshot.bindings,
        )
        authorization_ref = self.store.put_json(
            authorization, media_type=STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE
        )
        return self._append(
            kind=StudyEventKind.SEALED_AUTHORIZED,
            previous_tail_ref=previous_tail_ref,
            sealed_authorization_ref=authorization_ref,
            reason="sealed access authorized after complete study barrier",
        )

    def _reverify_all_live_runs(self, snapshot: StudySnapshot) -> None:
        """Fail closed unless every registered controller still owns its exact tail."""

        required = frozenset(_key_order(binding.key) for binding in snapshot.bindings)
        if frozenset(self._live_runs) != required:
            raise StudyBarrierError(
                "sealed authorization requires every process-local live run capability"
            )
        for binding in snapshot.bindings:
            key_tuple = _key_order(binding.key)
            live = self._live_runs[key_tuple]
            closure = self.journal._verify_run_closure(binding.run_closure_ref, binding.key)
            if (
                type(live.search_controller) is not SearchController
                or type(live.experiment_controller) is not ExperimentController
            ):
                raise StudyBarrierError("a registered live controller capability changed type")
            if (
                live.search_selection_tail_ref != closure.search_selection_tail_ref
                or live.experiment_selection_tail_ref != closure.experiment_selection_tail_ref
            ):
                raise StudyBarrierError("a registered live controller tail changed identity")
            try:
                search_snapshot = SearchController.verify_selection_closure(
                    live.search_controller, live.search_selection_tail_ref
                )
            except Exception as exc:
                raise StudyBarrierError(
                    f"live search selection re-verification failed: {exc}"
                ) from exc
            try:
                experiment_closure = ExperimentController.verify_experiment_selection_closure(
                    live.experiment_controller,
                    live.experiment_selection_tail_ref,
                )
            except Exception as exc:
                raise StudyBarrierError(
                    f"live experiment selection re-verification failed: {exc}"
                ) from exc
            try:
                if type(live.automatic_search_loop) is not AutomaticSearchLoop:
                    raise TypeError("automatic-search capability changed type")
                automatic_result = AutomaticSearchLoop.verify_execution(
                    live.automatic_search_loop,
                    live.automatic_search_result_ref,
                    search_run_ref=closure.search_run_ref,
                    controller=live.search_controller,
                    search_selection_tail_ref=live.search_selection_tail_ref,
                    lifecycle=live.experiment_controller,
                    experiment_selection_tail_ref=live.experiment_selection_tail_ref,
                )
            except Exception as exc:
                raise StudyBarrierError(
                    f"live automatic search re-verification failed: {exc}"
                ) from exc

            search_coordinates = (
                search_snapshot.tail_ref,
                search_snapshot.controller_manifest_ref,
                search_snapshot.search_run_ref,
                search_snapshot.selection_closure_ref,
                search_snapshot.baseline_kind,
                search_snapshot.search_seed,
                search_snapshot.champion_harness_ref,
                search_snapshot.champion_candidate_ref,
                search_snapshot.state,
            )
            expected_search_coordinates = (
                closure.search_selection_tail_ref,
                closure.search_controller_manifest_ref,
                closure.search_run_ref,
                closure.search_selection_closure_ref,
                closure.key.baseline_kind,
                closure.key.search_run_seed,
                closure.champion_harness_ref,
                closure.champion_candidate_ref,
                SearchState.SELECTION_CLOSED,
            )
            if search_coordinates != expected_search_coordinates:
                raise StudyBarrierError(
                    "live search selection no longer matches its registered closure"
                )

            stored_experiment_closure = self.store.get_json(
                closure.experiment_selection_closure_ref, SelectionClosure
            )
            if experiment_closure != stored_experiment_closure:
                raise StudyBarrierError(
                    "live experiment selection no longer matches its registered closure"
                )
            if (
                live.automatic_search_result_ref != closure.automatic_search_result_ref
                or automatic_result.admission_report_ref != closure.search_run_admission_report_ref
            ):
                raise StudyBarrierError(
                    "live automatic search no longer matches its registered closure"
                )

    def verify_sealed_authorization(self, tail_ref: ArtifactRef) -> StudySealedAuthorization:
        validated = ArtifactRef.model_validate(tail_ref)
        if validated != self._tail_ref:
            raise StaleStudyTailError("study sealed tail is stale or foreign")
        snapshot = self.journal.replay(validated)
        if snapshot.state is not StudyState.SEALED_AUTHORIZED:
            raise StudyBarrierError("study tail is not SEALED_AUTHORIZED")
        if snapshot.sealed_authorization_ref is None:  # pragma: no cover - replay enforces
            raise StudyJournalIntegrityError("sealed authorization artifact is missing")
        self._snapshot = snapshot
        return self.store.get_json(snapshot.sealed_authorization_ref, StudySealedAuthorization)

    def invalidate(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        invalidation_ref: ArtifactRef,
        reason: str,
    ) -> ArtifactRef:
        snapshot = self._require_tail(previous_tail_ref)
        if snapshot.state in {StudyState.INVALIDATED, StudyState.SEALED_AUTHORIZED}:
            raise StudyControllerError("terminal study state cannot be invalidated")
        checked_ref = ArtifactRef.model_validate(invalidation_ref)
        self.store.get_json(checked_ref)
        return self._append(
            kind=StudyEventKind.INVALIDATED,
            previous_tail_ref=previous_tail_ref,
            invalidation_ref=checked_ref,
            reason=self._normalize_reason(reason),
        )

    def _append(
        self,
        *,
        kind: StudyEventKind,
        previous_tail_ref: ArtifactRef | None,
        reason: str,
        run_key: StudyRunKey | None = None,
        run_closure_ref: ArtifactRef | None = None,
        barrier_closure_ref: ArtifactRef | None = None,
        sealed_authorization_ref: ArtifactRef | None = None,
        invalidation_ref: ArtifactRef | None = None,
    ) -> ArtifactRef:
        snapshot = self._snapshot
        sequence = 0 if snapshot is None else snapshot.sequence + 1
        if kind is StudyEventKind.INVALIDATED:
            from_state = None if snapshot is None else snapshot.state
            to_state = StudyState.INVALIDATED
        else:
            from_state, to_state = _EVENT_TRANSITIONS[kind]
        content = StudyJournalEventContent(
            study_controller_manifest_ref=self.study_controller_manifest_ref,
            baseline_study_plan_ref=self.manifest.baseline_study_plan_ref,
            plan_fingerprint=self.plan.fingerprint,
            study_id=self.manifest.study_id,
            sequence=sequence,
            previous_tail_ref=previous_tail_ref,
            kind=kind,
            from_state=from_state,
            to_state=to_state,
            run_key=run_key,
            run_closure_ref=run_closure_ref,
            barrier_closure_ref=barrier_closure_ref,
            sealed_authorization_ref=sealed_authorization_ref,
            invalidation_ref=invalidation_ref,
            reason=reason,
        )
        event = TrustedStudyEventService._issue(self.event_service, content)
        tail_ref = self.store.put_json(event, media_type=STUDY_JOURNAL_EVENT_MEDIA_TYPE)
        replayed = self.journal.replay(tail_ref)
        self._tail_ref = tail_ref
        self._snapshot = replayed
        return tail_ref

    def _require_tail(self, previous_tail_ref: ArtifactRef) -> StudySnapshot:
        validated = ArtifactRef.model_validate(previous_tail_ref)
        if validated.media_type != STUDY_JOURNAL_EVENT_MEDIA_TYPE:
            raise StaleStudyTailError("previous study tail declares the wrong media type")
        if self._tail_ref is None or validated != self._tail_ref:
            raise StaleStudyTailError("previous study tail is stale or foreign")
        snapshot = self.journal.replay(validated)
        self._snapshot = snapshot
        return snapshot

    @staticmethod
    def _require_state(snapshot: StudySnapshot, expected: StudyState) -> None:
        if snapshot.state is not expected:
            raise StudyControllerError(
                f"expected study state {expected.name}, got {snapshot.state.name}"
            )

    @staticmethod
    def _normalize_reason(reason: str) -> str:
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        normalized = reason.strip()
        if not normalized:
            raise ValueError("reason must not be empty")
        return normalized


__all__ = [
    "STUDY_BARRIER_CLOSURE_MEDIA_TYPE",
    "STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE",
    "STUDY_JOURNAL_EVENT_MEDIA_TYPE",
    "STUDY_RUN_CLOSURE_MEDIA_TYPE",
    "STUDY_SEALED_AUTHORIZATION_MEDIA_TYPE",
    "StaleStudyTailError",
    "StudyBarrierClosure",
    "StudyBarrierError",
    "StudyController",
    "StudyControllerError",
    "StudyControllerManifest",
    "StudyEventAttestationError",
    "StudyEventKind",
    "StudyEventVerificationCapability",
    "StudyExpectedRunManifest",
    "StudyJournal",
    "StudyJournalCycleError",
    "StudyJournalEvent",
    "StudyJournalEventContent",
    "StudyJournalIntegrityError",
    "StudyRunBinding",
    "StudyRunClosure",
    "StudyRunKey",
    "StudySealedAuthorization",
    "StudySnapshot",
    "StudyState",
    "TrustedStudyEventService",
    "expected_run_keys",
]
