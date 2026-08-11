"""Attested, append-only control for one baseline arm and one search run.

The controller in this module owns *search* semantics only.  Candidate
admission and gate decisions remain owned by :class:`ExperimentController`.
The two layers meet through a verify-only ``TerminalAuthorizationVerifier``:
being able to place a plausible JSON object in the artifact store is not
enough to advance the search champion.

Journal events are content-addressed linked tails.  Every event repeats the
frozen run coordinates, its sequence, and its previous tail, and is attested
with a process-scoped HMAC capability.  Durable multi-process signing is a
later service concern; replay only needs the paired verify capability plus the
terminal-authorization verifier.

``TerminalAuthorizationVerifier`` is an explicit injection trust boundary for
standalone search use, not a nominal capability type.  Production study
orchestration supplies a concrete :class:`ExperimentController` and rechecks
it at the study barrier.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from enum import StrEnum
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.experiment import EXPERIMENT_MANIFEST_MEDIA_TYPE
from spiral_harness.core.lifecycle import CandidateState
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.evolution.models import (
    SearchPolicy,
    SearchRunManifest,
    SearchStoppingPolicy,
)
from spiral_harness.experiments.baselines import BaselineKind
from spiral_harness.experiments.controller import (
    TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE,
    TerminalTransitionAuthorization,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.models import Decision, GateCheckOutcome, GateDecision

SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE = (
    "application/vnd.spiral-harness.search-controller-manifest.v1+json"
)
SEARCH_RUN_MANIFEST_MEDIA_TYPE = "application/vnd.spiral-harness.search-run-manifest.v1+json"
SEARCH_JOURNAL_EVENT_MEDIA_TYPE = "application/vnd.spiral-harness.search-journal-event.v1+json"
AGGREGATE_FEEDBACK_DISCLOSURE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.aggregate-feedback-disclosure.v1+json"
)
SEARCH_SELECTION_CLOSURE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.search-selection-closure.v1+json"
)

_EVENT_ATTESTATION_DOMAIN = b"spiral-harness:search-journal-event-attestation:v1\x00"
_EVENT_ATTESTOR_ID_DOMAIN = b"spiral-harness:search-journal-event-attestor-id:v1\x00"


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


def _canonical_refs(refs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
    return tuple(sorted(refs, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))


class SearchControllerError(RuntimeError):
    """Base error for rejected search-control operations."""


class SearchJournalIntegrityError(SearchControllerError):
    """Raised when an attested journal is malformed or foreign."""


class SearchJournalCycleError(SearchJournalIntegrityError):
    """Raised when a linked journal revisits a tail reference."""


class SearchEventAttestationError(SearchJournalIntegrityError):
    """Raised when a search event is not authentic for the frozen signer."""


class StaleSearchTailError(SearchControllerError):
    """Raised when a mutation is requested from an old or foreign tail."""


class SearchStopError(SearchControllerError):
    """Raised when work is attempted after a frozen stop criterion is met."""


class SearchState(StrEnum):
    """Trusted states of one arm/run search controller."""

    FROZEN = "frozen"
    SEARCHING = "searching"
    SELECTION_CLOSED = "selection_closed"
    INVALIDATED = "invalidated"


class RoundStage(StrEnum):
    """Strict within-round progression."""

    ROUND_OPEN = "round_open"
    EVIDENCE_READY = "evidence_ready"
    CANDIDATE_POOL_READY = "candidate_pool_ready"
    SCREENED = "screened"
    NOMINATED = "nominated"
    NO_CANDIDATE = "no_candidate"
    ROUND_COMPLETE = "round_complete"


class SearchEventKind(StrEnum):
    """Semantic operation represented by a journal event."""

    FROZEN = "frozen"
    SEARCH_STARTED = "search_started"
    ROUND_OPENED = "round_opened"
    EVIDENCE_RECORDED = "evidence_recorded"
    CANDIDATE_POOL_RECORDED = "candidate_pool_recorded"
    SCREEN_RECORDED = "screen_recorded"
    CANDIDATE_NOMINATED = "candidate_nominated"
    NO_CANDIDATE_RECORDED = "no_candidate_recorded"
    NOMINATED_ROUND_COMPLETED = "nominated_round_completed"
    EMPTY_ROUND_COMPLETED = "empty_round_completed"
    SELECTION_CLOSED = "selection_closed"
    INVALIDATED = "invalidated"


class SearchStopReason(StrEnum):
    """Replay-derived reasons that authorize closing selection."""

    STATIC = "static"
    MAX_ROUNDS = "max_rounds"
    MAX_GATE_NOMINATIONS = "max_gate_nominations"
    PATIENCE = "patience"
    MAX_CONSECUTIVE_DECLINES = "max_consecutive_declines"


class SearchControllerManifest(ImmutableModel):
    """Freeze one controller journal before any search operation.

    ``search_run_ref`` points at the higher-level typed search-run manifest.
    The identity coordinates are intentionally repeated here and in every
    event so a journal can be rejected without importing optimizer models.
    """

    schema_version: Literal["1"] = "1"
    search_run_ref: ArtifactRef
    study_ref: ArtifactRef
    experiment_ref: ArtifactRef
    run_id: NonEmptyStr
    baseline_kind: BaselineKind
    search_seed: Annotated[int, Field(ge=0, strict=True)]
    initial_champion_harness_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    max_rounds: Annotated[int, Field(ge=0, strict=True)]
    max_gate_nominations: Annotated[int, Field(ge=0, strict=True)]
    patience_rounds: Annotated[int, Field(ge=0, strict=True)]
    max_consecutive_declines: Annotated[int, Field(ge=0, strict=True)]
    journal_attestor_id: Sha256

    @model_validator(mode="after")
    def refs_and_static_limits_are_valid(self) -> SearchControllerManifest:
        for field_name in (
            "study_ref",
            "initial_champion_harness_ref",
            "analysis_plan_ref",
        ):
            _require_json_ref(getattr(self, field_name), field_name=field_name)
        if self.search_run_ref.media_type != SEARCH_RUN_MANIFEST_MEDIA_TYPE:
            raise ValueError("search_run_ref declares the wrong search-run manifest media type")
        if self.experiment_ref.media_type != EXPERIMENT_MANIFEST_MEDIA_TYPE:
            raise ValueError("experiment_ref declares the wrong experiment manifest media type")
        if self.baseline_kind is BaselineKind.STATIC:
            if any(
                (
                    self.max_rounds,
                    self.max_gate_nominations,
                    self.patience_rounds,
                    self.max_consecutive_declines,
                )
            ):
                raise ValueError("static search must freeze all stopping counts at zero")
        elif any(
            value == 0
            for value in (
                self.max_rounds,
                self.max_gate_nominations,
                self.patience_rounds,
                self.max_consecutive_declines,
            )
        ):
            raise ValueError("non-static search must freeze positive stopping counts")
        return self


class AggregateGateCheckOutcome(ImmutableModel):
    """Redacted outcome of one trusted, fixed-name gate check."""

    name: Literal[
        "integrity",
        "sample_size",
        "policy",
        "mechanism",
        "primary_effect",
        "protected_slices",
        "task_regressions",
        "resources",
    ]
    outcome: GateCheckOutcome


_AGGREGATE_GATE_CHECK_NAMES = (
    "integrity",
    "sample_size",
    "policy",
    "mechanism",
    "primary_effect",
    "protected_slices",
    "task_regressions",
    "resources",
)


class AggregateFeedbackSummary(ImmutableModel):
    """Fixed-shape aggregate-only feedback safe to disclose to an optimizer.

    There are deliberately no task identifiers, item arrays, trajectories, or
    free-form artifact references in this model.
    """

    decision: Decision
    gate_version: NonEmptyStr
    gate_config_sha256: Sha256
    checks: tuple[AggregateGateCheckOutcome, ...]
    n_valid_pairs: Annotated[int, Field(ge=0, strict=True)]
    n_tasks: Annotated[int, Field(ge=0, strict=True)]
    mean_score_delta: Annotated[float, Field(allow_inf_nan=False)] | None
    confidence_lower_bound: Annotated[float, Field(allow_inf_nan=False)] | None = None
    confidence_upper_bound: Annotated[float, Field(allow_inf_nan=False)] | None = None
    confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = None
    regression_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    tokens_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    latency_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    tool_calls_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def check_names_are_unique(self) -> AggregateFeedbackSummary:
        names = tuple(check.name for check in self.checks)
        if names != _AGGREGATE_GATE_CHECK_NAMES:
            raise ValueError("aggregate gate checks must be the complete canonical set")
        ci_values = (
            self.confidence_lower_bound,
            self.confidence_upper_bound,
            self.confidence_level,
        )
        if any(value is None for value in ci_values) != all(value is None for value in ci_values):
            raise ValueError("confidence interval aggregate fields must be present together")
        return self


class AggregateFeedbackDisclosure(ImmutableModel):
    """Controller-authored aggregate view; raw gate comparisons never enter it."""

    schema_version: Literal["1"] = "1"
    controller_manifest_ref: ArtifactRef
    search_run_ref: ArtifactRef
    study_ref: ArtifactRef
    experiment_ref: ArtifactRef
    run_id: NonEmptyStr
    baseline_kind: BaselineKind
    search_seed: Annotated[int, Field(ge=0, strict=True)]
    round_index: Annotated[int, Field(ge=0, strict=True)]
    candidate_ref: ArtifactRef
    terminal_authorization_ref: ArtifactRef
    terminal_state: CandidateState
    champion_harness_ref: ArtifactRef
    gate_version: NonEmptyStr
    gate_config_sha256: Sha256
    summary: AggregateFeedbackSummary

    @model_validator(mode="after")
    def exact_reference_types_and_gate_state(self) -> AggregateFeedbackDisclosure:
        if self.controller_manifest_ref.media_type != SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE:
            raise ValueError("controller_manifest_ref declares the wrong media type")
        if (
            self.terminal_authorization_ref.media_type
            != TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE
        ):
            raise ValueError("terminal_authorization_ref declares the wrong media type")
        if self.terminal_state not in {
            CandidateState.PROMOTED,
            CandidateState.REJECTED,
            CandidateState.INCONCLUSIVE,
        }:
            raise ValueError("aggregate feedback requires a gate terminal state")
        if (
            self.gate_version != self.summary.gate_version
            or self.gate_config_sha256 != self.summary.gate_config_sha256
        ):
            raise ValueError("disclosure gate provenance must match its summary")
        return self


class SearchSelectionClosure(ImmutableModel):
    """Controller-authored selection closure using the pre-frozen analysis plan."""

    schema_version: Literal["1"] = "1"
    controller_manifest_ref: ArtifactRef
    search_run_ref: ArtifactRef
    study_ref: ArtifactRef
    experiment_ref: ArtifactRef
    run_id: NonEmptyStr
    baseline_kind: BaselineKind
    search_seed: Annotated[int, Field(ge=0, strict=True)]
    search_tail_before_close_ref: ArtifactRef
    champion_harness_ref: ArtifactRef
    champion_candidate_ref: ArtifactRef | None
    analysis_plan_ref: ArtifactRef
    stop_reasons: Annotated[tuple[SearchStopReason, ...], Field(min_length=1)]
    completed_rounds: Annotated[int, Field(ge=0, strict=True)]
    gate_nominations: Annotated[int, Field(ge=0, strict=True)]

    @field_validator("stop_reasons")
    @classmethod
    def canonicalize_stop_reasons(
        cls, reasons: tuple[SearchStopReason, ...]
    ) -> tuple[SearchStopReason, ...]:
        ordered = tuple(sorted(reasons, key=lambda reason: reason.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("stop_reasons must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def exact_manifest_and_tail_media_types(self) -> SearchSelectionClosure:
        if self.controller_manifest_ref.media_type != SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE:
            raise ValueError("controller_manifest_ref declares the wrong media type")
        if self.search_tail_before_close_ref.media_type != SEARCH_JOURNAL_EVENT_MEDIA_TYPE:
            raise ValueError("search_tail_before_close_ref declares the wrong media type")
        return self


class SearchJournalEventContent(ImmutableModel):
    """Unsigned, exhaustively shaped content of one linked search event."""

    schema_version: Literal["1"] = "1"
    controller_manifest_ref: ArtifactRef
    search_run_ref: ArtifactRef
    study_ref: ArtifactRef
    experiment_ref: ArtifactRef
    run_id: NonEmptyStr
    baseline_kind: BaselineKind
    search_seed: Annotated[int, Field(ge=0, strict=True)]
    sequence: Annotated[int, Field(ge=0, strict=True)]
    previous_tail_ref: ArtifactRef | None
    kind: SearchEventKind
    from_state: SearchState | None
    to_state: SearchState
    round_index: Annotated[int, Field(ge=0, strict=True)] | None = None
    from_stage: RoundStage | None = None
    to_stage: RoundStage | None = None
    champion_before_ref: ArtifactRef
    champion_after_ref: ArtifactRef
    evidence_packet_ref: ArtifactRef | None = None
    candidate_pool_ref: ArtifactRef | None = None
    candidate_refs: tuple[ArtifactRef, ...] = ()
    proposal_declined: bool = False
    screen_report_ref: ArtifactRef | None = None
    eligible_candidate_refs: tuple[ArtifactRef, ...] = ()
    nomination_ref: ArtifactRef | None = None
    nominated_candidate_ref: ArtifactRef | None = None
    terminal_authorization_ref: ArtifactRef | None = None
    feedback_disclosure_ref: ArtifactRef | None = None
    selection_closure_ref: ArtifactRef | None = None
    invalidation_ref: ArtifactRef | None = None
    stop_reasons: tuple[SearchStopReason, ...] = ()
    reason: NonEmptyStr

    @field_validator("candidate_refs", "eligible_candidate_refs")
    @classmethod
    def canonicalize_candidate_refs(cls, refs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
        ordered = _canonical_refs(refs)
        if len({ref.sha256 for ref in ordered}) != len(ordered):
            raise ValueError("candidate refs must not contain duplicate artifacts")
        return ordered

    @field_validator("stop_reasons")
    @classmethod
    def canonicalize_event_stop_reasons(
        cls, reasons: tuple[SearchStopReason, ...]
    ) -> tuple[SearchStopReason, ...]:
        ordered = tuple(sorted(reasons, key=lambda reason: reason.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("stop_reasons must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def linked_shape_and_event_payload_are_exact(self) -> SearchJournalEventContent:
        if self.controller_manifest_ref.media_type != SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE:
            raise ValueError("controller_manifest_ref declares the wrong media type")
        if self.sequence == 0 and self.previous_tail_ref is not None:
            raise ValueError("search sequence 0 must not have a previous tail")
        if self.sequence > 0 and self.previous_tail_ref is None:
            raise ValueError("search sequence greater than 0 requires a previous tail")
        if (
            self.previous_tail_ref is not None
            and self.previous_tail_ref.media_type != SEARCH_JOURNAL_EVENT_MEDIA_TYPE
        ):
            raise ValueError("previous_tail_ref declares the wrong media type")
        for ref in (
            *self.candidate_refs,
            *self.eligible_candidate_refs,
            self.evidence_packet_ref,
            self.candidate_pool_ref,
            self.screen_report_ref,
            self.nomination_ref,
            self.nominated_candidate_ref,
            self.invalidation_ref,
        ):
            if ref is not None:
                _require_json_ref(ref, field_name="event artifact ref")
        exact_optional_media = (
            (
                self.terminal_authorization_ref,
                TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE,
                "terminal_authorization_ref",
            ),
            (
                self.feedback_disclosure_ref,
                AGGREGATE_FEEDBACK_DISCLOSURE_MEDIA_TYPE,
                "feedback_disclosure_ref",
            ),
            (
                self.selection_closure_ref,
                SEARCH_SELECTION_CLOSURE_MEDIA_TYPE,
                "selection_closure_ref",
            ),
        )
        for ref, media_type, field_name in exact_optional_media:
            if ref is not None and ref.media_type != media_type:
                raise ValueError(f"{field_name} declares the wrong media type")

        actual_transition = (
            self.from_state,
            self.to_state,
            self.from_stage,
            self.to_stage,
        )
        if self.kind is SearchEventKind.INVALIDATED:
            if (
                self.from_state
                not in {
                    SearchState.FROZEN,
                    SearchState.SEARCHING,
                    SearchState.SELECTION_CLOSED,
                }
                or self.to_state is not SearchState.INVALIDATED
                or self.to_stage is not None
            ):
                raise ValueError("invalidated declares an invalid state/stage transition")
        else:
            expected = _EVENT_SHAPES[self.kind]
            if actual_transition != expected:
                raise ValueError(f"{self.kind.value} declares an invalid state/stage transition")

        optional_payloads = {
            "evidence_packet_ref": self.evidence_packet_ref,
            "candidate_pool_ref": self.candidate_pool_ref,
            "screen_report_ref": self.screen_report_ref,
            "nomination_ref": self.nomination_ref,
            "nominated_candidate_ref": self.nominated_candidate_ref,
            "terminal_authorization_ref": self.terminal_authorization_ref,
            "feedback_disclosure_ref": self.feedback_disclosure_ref,
            "selection_closure_ref": self.selection_closure_ref,
            "invalidation_ref": self.invalidation_ref,
        }
        required_fields = _EVENT_REQUIRED_REFS[self.kind]
        allowed_fields = _EVENT_ALLOWED_REFS[self.kind]
        for field_name, value in optional_payloads.items():
            if field_name in required_fields and value is None:
                raise ValueError(f"{self.kind.value} requires {field_name}")
            if field_name not in allowed_fields and value is not None:
                raise ValueError(f"{self.kind.value} must not include {field_name}")

        round_kinds = {
            SearchEventKind.ROUND_OPENED,
            SearchEventKind.EVIDENCE_RECORDED,
            SearchEventKind.CANDIDATE_POOL_RECORDED,
            SearchEventKind.SCREEN_RECORDED,
            SearchEventKind.CANDIDATE_NOMINATED,
            SearchEventKind.NO_CANDIDATE_RECORDED,
            SearchEventKind.NOMINATED_ROUND_COMPLETED,
            SearchEventKind.EMPTY_ROUND_COMPLETED,
        }
        if self.kind is SearchEventKind.INVALIDATED:
            if (self.round_index is None) != (self.from_stage is None):
                raise ValueError("active-round invalidation must bind round_index and from_stage")
        elif (self.round_index is not None) != (self.kind in round_kinds):
            raise ValueError("round_index presence does not match event kind")
        if self.kind is SearchEventKind.CANDIDATE_POOL_RECORDED:
            if self.proposal_declined and self.candidate_refs:
                raise ValueError("a declined proposal cannot publish candidates")
        elif self.candidate_refs or self.proposal_declined:
            raise ValueError("candidate pool payload is only valid on candidate_pool_recorded")
        if self.kind is not SearchEventKind.SCREEN_RECORDED and self.eligible_candidate_refs:
            raise ValueError("eligible candidates are only valid on screen_recorded")
        if self.kind is SearchEventKind.SELECTION_CLOSED:
            if not self.stop_reasons:
                raise ValueError("selection closure requires stop_reasons")
        elif self.stop_reasons:
            raise ValueError("stop_reasons are only valid on selection closure")
        if (
            self.kind is not SearchEventKind.NOMINATED_ROUND_COMPLETED
            and self.champion_after_ref != self.champion_before_ref
        ):
            raise ValueError("only a nominated terminal event may advance champion")
        return self


_EVENT_SHAPES: dict[
    SearchEventKind,
    tuple[SearchState | None, SearchState, RoundStage | None, RoundStage | None],
] = {
    SearchEventKind.FROZEN: (None, SearchState.FROZEN, None, None),
    SearchEventKind.SEARCH_STARTED: (
        SearchState.FROZEN,
        SearchState.SEARCHING,
        None,
        None,
    ),
    SearchEventKind.ROUND_OPENED: (
        SearchState.SEARCHING,
        SearchState.SEARCHING,
        None,
        RoundStage.ROUND_OPEN,
    ),
    SearchEventKind.EVIDENCE_RECORDED: (
        SearchState.SEARCHING,
        SearchState.SEARCHING,
        RoundStage.ROUND_OPEN,
        RoundStage.EVIDENCE_READY,
    ),
    SearchEventKind.CANDIDATE_POOL_RECORDED: (
        SearchState.SEARCHING,
        SearchState.SEARCHING,
        RoundStage.EVIDENCE_READY,
        RoundStage.CANDIDATE_POOL_READY,
    ),
    SearchEventKind.SCREEN_RECORDED: (
        SearchState.SEARCHING,
        SearchState.SEARCHING,
        RoundStage.CANDIDATE_POOL_READY,
        RoundStage.SCREENED,
    ),
    SearchEventKind.CANDIDATE_NOMINATED: (
        SearchState.SEARCHING,
        SearchState.SEARCHING,
        RoundStage.SCREENED,
        RoundStage.NOMINATED,
    ),
    SearchEventKind.NO_CANDIDATE_RECORDED: (
        SearchState.SEARCHING,
        SearchState.SEARCHING,
        RoundStage.SCREENED,
        RoundStage.NO_CANDIDATE,
    ),
    SearchEventKind.NOMINATED_ROUND_COMPLETED: (
        SearchState.SEARCHING,
        SearchState.SEARCHING,
        RoundStage.NOMINATED,
        RoundStage.ROUND_COMPLETE,
    ),
    SearchEventKind.EMPTY_ROUND_COMPLETED: (
        SearchState.SEARCHING,
        SearchState.SEARCHING,
        RoundStage.NO_CANDIDATE,
        RoundStage.ROUND_COMPLETE,
    ),
    SearchEventKind.SELECTION_CLOSED: (
        SearchState.SEARCHING,
        SearchState.SELECTION_CLOSED,
        None,
        None,
    ),
    # Invalidation may occur while a round is active.  The controller replaces
    # this placeholder with exact current stages before model validation; the
    # model validator special-cases it below in ``_validate_invalidation_shape``.
    SearchEventKind.INVALIDATED: (SearchState.SEARCHING, SearchState.INVALIDATED, None, None),
}

_EVENT_REQUIRED_REFS: dict[SearchEventKind, frozenset[str]] = {
    kind: frozenset() for kind in SearchEventKind
}
_EVENT_REQUIRED_REFS.update(
    {
        SearchEventKind.EVIDENCE_RECORDED: frozenset({"evidence_packet_ref"}),
        SearchEventKind.CANDIDATE_POOL_RECORDED: frozenset({"candidate_pool_ref"}),
        SearchEventKind.SCREEN_RECORDED: frozenset({"screen_report_ref"}),
        SearchEventKind.CANDIDATE_NOMINATED: frozenset(
            {"nomination_ref", "nominated_candidate_ref"}
        ),
        SearchEventKind.NOMINATED_ROUND_COMPLETED: frozenset(
            {
                "nominated_candidate_ref",
                "terminal_authorization_ref",
                "feedback_disclosure_ref",
            }
        ),
        SearchEventKind.SELECTION_CLOSED: frozenset({"selection_closure_ref"}),
        SearchEventKind.INVALIDATED: frozenset({"invalidation_ref"}),
    }
)
_EVENT_ALLOWED_REFS = dict(_EVENT_REQUIRED_REFS)


class SearchJournalEvent(SearchJournalEventContent):
    """One HMAC-attested linked event."""

    attestor_id: Sha256
    attestation_sha256: Sha256

    @property
    def content(self) -> SearchJournalEventContent:
        return SearchJournalEventContent.model_validate(
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


def _attestation(secret: bytes, content: SearchJournalEventContent, attestor_id: str) -> str:
    return hmac.new(
        secret,
        _EVENT_ATTESTATION_DOMAIN
        + attestor_id.encode("ascii")
        + b"\x00"
        + canonical_json_bytes(content),
        hashlib.sha256,
    ).hexdigest()


class SearchEventVerificationCapability:
    """Verify events from one process-scoped trusted search controller."""

    __slots__ = ("__secret", "_attestor_id")

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("search event verification secret must contain at least 32 bytes")
        self.__secret = secret
        self._attestor_id = _attestor_id(secret)

    @property
    def attestor_id(self) -> str:
        return self._attestor_id

    def verify(self, event: SearchJournalEvent | object) -> SearchJournalEvent:
        validated = SearchJournalEvent.model_validate(event)
        if validated.attestor_id != self._attestor_id:
            raise SearchEventAttestationError("search event was signed by a foreign attestor")
        expected = _attestation(self.__secret, validated.content, self._attestor_id)
        if not hmac.compare_digest(validated.attestation_sha256, expected):
            raise SearchEventAttestationError("search event HMAC attestation is invalid")
        return validated


class TrustedSearchEventService:
    """Private signing capability retained by trusted search orchestration."""

    __slots__ = ("__secret", "_verification_capability")

    def __init__(self, secret: bytes | None = None) -> None:
        material = secrets.token_bytes(32) if secret is None else secret
        if not isinstance(material, bytes) or len(material) < 32:
            raise ValueError("search event signing secret must contain at least 32 bytes")
        self.__secret = material
        self._verification_capability = SearchEventVerificationCapability(material)

    @property
    def attestor_id(self) -> str:
        return self._verification_capability.attestor_id

    @property
    def verification_capability(self) -> SearchEventVerificationCapability:
        return self._verification_capability

    def _issue(self, content: SearchJournalEventContent) -> SearchJournalEvent:
        """Issue an event for :class:`SearchController`; never expose to workers."""

        validated = SearchJournalEventContent.model_validate(content)
        return SearchJournalEvent(
            **validated.model_dump(mode="python", round_trip=True, warnings="none"),
            attestor_id=self.attestor_id,
            attestation_sha256=_attestation(self.__secret, validated, self.attestor_id),
        )


@runtime_checkable
class TerminalAuthorizationVerifier(Protocol):
    """Trusted injection boundary that authenticates terminal artifacts.

    Runtime protocol conformance alone is not an authenticity proof.  Callers
    outside the study controller must root this object in trusted composition.
    """

    def verify_terminal_authorization(
        self, ref: ArtifactRef
    ) -> TerminalTransitionAuthorization: ...


class SearchRunSnapshot(ImmutableModel):
    """Replay-derived state at one exact journal tail."""

    controller_manifest_ref: ArtifactRef
    search_run_ref: ArtifactRef
    study_ref: ArtifactRef
    experiment_ref: ArtifactRef
    run_id: NonEmptyStr
    baseline_kind: BaselineKind
    search_seed: Annotated[int, Field(ge=0, strict=True)]
    tail_ref: ArtifactRef
    sequence: Annotated[int, Field(ge=0, strict=True)]
    event_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    state: SearchState
    active_round_index: Annotated[int, Field(ge=0, strict=True)] | None
    round_stage: RoundStage | None
    completed_rounds: Annotated[int, Field(ge=0, strict=True)]
    gate_nominations: Annotated[int, Field(ge=0, strict=True)]
    rounds_since_promotion: Annotated[int, Field(ge=0, strict=True)]
    consecutive_declines: Annotated[int, Field(ge=0, strict=True)]
    champion_harness_ref: ArtifactRef
    champion_candidate_ref: ArtifactRef | None
    candidate_refs: tuple[ArtifactRef, ...]
    eligible_candidate_refs: tuple[ArtifactRef, ...]
    nomination_ref: ArtifactRef | None
    nominated_candidate_ref: ArtifactRef | None
    current_proposal_declined: bool
    terminal_authorization_refs: tuple[ArtifactRef, ...]
    feedback_disclosure_refs: tuple[ArtifactRef, ...]
    last_feedback_disclosure_ref: ArtifactRef | None
    selection_closure_ref: ArtifactRef | None
    stop_reasons: tuple[SearchStopReason, ...]
    proposal_api_open: bool


def _stop_reasons(
    manifest: SearchControllerManifest,
    *,
    completed_rounds: int,
    gate_nominations: int,
    rounds_since_promotion: int,
    consecutive_declines: int,
) -> tuple[SearchStopReason, ...]:
    if manifest.baseline_kind is BaselineKind.STATIC:
        return (SearchStopReason.STATIC,)
    reasons: list[SearchStopReason] = []
    if completed_rounds >= manifest.max_rounds:
        reasons.append(SearchStopReason.MAX_ROUNDS)
    if gate_nominations >= manifest.max_gate_nominations:
        reasons.append(SearchStopReason.MAX_GATE_NOMINATIONS)
    if completed_rounds > 0 and rounds_since_promotion >= manifest.patience_rounds:
        reasons.append(SearchStopReason.PATIENCE)
    if consecutive_declines >= manifest.max_consecutive_declines:
        reasons.append(SearchStopReason.MAX_CONSECUTIVE_DECLINES)
    return tuple(sorted(reasons, key=lambda reason: reason.value))


class SearchJournal:
    """Read and verify one immutable linked search-run journal."""

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        controller_manifest_ref: ArtifactRef,
        event_verifier: SearchEventVerificationCapability,
        terminal_authorization_verifier: TerminalAuthorizationVerifier | None = None,
    ) -> None:
        if type(event_verifier) is not SearchEventVerificationCapability:
            raise TypeError("event_verifier must be an exact SearchEventVerificationCapability")
        self.store = store
        self.controller_manifest_ref = ArtifactRef.model_validate(controller_manifest_ref)
        if self.controller_manifest_ref.media_type != SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE:
            raise SearchJournalIntegrityError(
                "controller manifest ref declares the wrong media type"
            )
        self.manifest = self.store.get_json(self.controller_manifest_ref, SearchControllerManifest)
        search_run = self.store.get_json(self.manifest.search_run_ref, SearchRunManifest)
        expected_run_coordinates = (
            self.manifest.study_ref,
            self.manifest.experiment_ref,
            self.manifest.baseline_kind,
            self.manifest.search_seed,
            self.manifest.initial_champion_harness_ref,
            self.manifest.analysis_plan_ref,
        )
        actual_run_coordinates = (
            search_run.baseline_study_plan_ref,
            search_run.experiment_ref,
            search_run.baseline_kind,
            search_run.search_run_seed,
            search_run.seed_harness_ref,
            search_run.analysis_plan_ref,
        )
        if actual_run_coordinates != expected_run_coordinates:
            raise SearchJournalIntegrityError(
                "controller manifest coordinates do not match search_run_ref"
            )
        for ref, label in (
            (self.manifest.study_ref, "study"),
            (self.manifest.experiment_ref, "experiment"),
            (self.manifest.initial_champion_harness_ref, "initial champion"),
            (self.manifest.analysis_plan_ref, "analysis plan"),
        ):
            try:
                self.store.get_json(ref)
            except Exception as exc:
                raise SearchJournalIntegrityError(
                    f"frozen {label} artifact could not be verified: {exc}"
                ) from exc
        search_policy = self.store.get_json(search_run.search_policy_ref, SearchPolicy)
        stopping_policy = self.store.get_json(search_run.stopping_policy_ref, SearchStoppingPolicy)
        if stopping_policy != search_policy.stopping_policy:
            raise SearchJournalIntegrityError(
                "search stopping policy does not match the frozen search policy"
            )
        if search_policy.baseline_kind is not self.manifest.baseline_kind:
            raise SearchJournalIntegrityError(
                "search policy baseline does not match controller manifest"
            )
        expected_limits = (
            search_policy.max_rounds,
            search_policy.max_gate_queries,
            search_policy.patience_rounds,
            search_policy.max_consecutive_declines,
        )
        actual_limits = (
            self.manifest.max_rounds,
            self.manifest.max_gate_nominations,
            self.manifest.patience_rounds,
            self.manifest.max_consecutive_declines,
        )
        if actual_limits != expected_limits:
            raise SearchJournalIntegrityError(
                "controller stop limits do not match the frozen search policy"
            )
        self.search_run = search_run
        self.search_policy = search_policy
        self.stopping_policy = stopping_policy
        if event_verifier.attestor_id != self.manifest.journal_attestor_id:
            raise SearchEventAttestationError(
                "event verifier does not match the controller manifest attestor"
            )
        self.event_verifier = event_verifier
        if self.manifest.baseline_kind is not BaselineKind.STATIC and (
            terminal_authorization_verifier is None
            or not isinstance(terminal_authorization_verifier, TerminalAuthorizationVerifier)
        ):
            raise SearchControllerError(
                "non-static search requires a terminal authorization verifier"
            )
        self.terminal_authorization_verifier = terminal_authorization_verifier

    def replay(self, tail_ref: ArtifactRef) -> SearchRunSnapshot:
        """Verify links, coordinates, HMACs, and semantic transitions."""

        validated_tail = ArtifactRef.model_validate(tail_ref)
        events_backwards: list[tuple[ArtifactRef, SearchJournalEvent]] = []
        seen: set[str] = set()
        cursor: ArtifactRef | None = validated_tail
        while cursor is not None:
            if cursor.media_type != SEARCH_JOURNAL_EVENT_MEDIA_TYPE:
                raise SearchJournalIntegrityError(
                    "search journal tail declares the wrong media type"
                )
            if cursor.sha256 in seen:
                raise SearchJournalCycleError(f"search journal cycle detected at {cursor.sha256}")
            seen.add(cursor.sha256)
            raw = self.store.get_json(cursor, SearchJournalEvent)
            event = SearchEventVerificationCapability.verify(self.event_verifier, raw)
            events_backwards.append((cursor, event))
            cursor = event.previous_tail_ref

        events = tuple(reversed(events_backwards))
        if not events:  # pragma: no cover - one tail always causes one read
            raise SearchJournalIntegrityError("search journal is empty")
        for expected_sequence, (ref, event) in enumerate(events):
            self._verify_coordinates(event)
            if event.sequence != expected_sequence:
                raise SearchJournalIntegrityError(
                    "search sequence is not contiguous from zero: "
                    f"expected {expected_sequence}, got {event.sequence}"
                )
            expected_previous = None if expected_sequence == 0 else events[expected_sequence - 1][0]
            if event.previous_tail_ref != expected_previous:
                raise SearchJournalIntegrityError(
                    f"search link is inconsistent at sequence {event.sequence}"
                )
            if ref.media_type != SEARCH_JOURNAL_EVENT_MEDIA_TYPE:  # pragma: no cover
                raise SearchJournalIntegrityError("search event ref media type changed")
        return self._replay_semantics(events)

    def _verify_coordinates(self, event: SearchJournalEvent) -> None:
        manifest = self.manifest
        expected = (
            self.controller_manifest_ref,
            manifest.search_run_ref,
            manifest.study_ref,
            manifest.experiment_ref,
            manifest.run_id,
            manifest.baseline_kind,
            manifest.search_seed,
            manifest.journal_attestor_id,
        )
        actual = (
            event.controller_manifest_ref,
            event.search_run_ref,
            event.study_ref,
            event.experiment_ref,
            event.run_id,
            event.baseline_kind,
            event.search_seed,
            event.attestor_id,
        )
        if actual != expected:
            raise SearchJournalIntegrityError(
                f"foreign search-run coordinates at sequence {event.sequence}"
            )

    def _verify_terminal_authorization(
        self,
        ref: ArtifactRef,
        *,
        candidate_ref: ArtifactRef,
        champion_ref: ArtifactRef,
    ) -> TerminalTransitionAuthorization:
        verifier = self.terminal_authorization_verifier
        if verifier is None:  # pragma: no cover - static cannot nominate
            raise SearchJournalIntegrityError("terminal authorization verifier is unavailable")
        if ref.media_type != TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE:
            raise SearchJournalIntegrityError(
                "terminal authorization ref declares the wrong media type"
            )
        try:
            authorization = TerminalTransitionAuthorization.model_validate(
                verifier.verify_terminal_authorization(ref)
            )
        except Exception as exc:
            raise SearchJournalIntegrityError(
                f"terminal authorization verification failed: {exc}"
            ) from exc
        try:
            stored_authorization = self.store.get_json(ref, TerminalTransitionAuthorization)
        except Exception as exc:
            raise SearchJournalIntegrityError(
                f"verified terminal authorization artifact could not be loaded: {exc}"
            ) from exc
        if stored_authorization != authorization:
            raise SearchJournalIntegrityError(
                "terminal verifier result does not match the referenced artifact"
            )
        if authorization.experiment_ref != self.manifest.experiment_ref:
            raise SearchJournalIntegrityError(
                "terminal authorization belongs to a foreign experiment"
            )
        if authorization.candidate_ref != candidate_ref:
            raise SearchJournalIntegrityError(
                "terminal authorization belongs to a foreign candidate"
            )
        if authorization.prior_champion_harness_ref != champion_ref:
            raise SearchJournalIntegrityError(
                "terminal authorization prior champion does not match search champion"
            )
        if authorization.parent_harness_ref != champion_ref:
            raise SearchJournalIntegrityError(
                "terminal authorization parent does not match search champion"
            )
        if authorization.terminal_state not in {
            CandidateState.PROMOTED,
            CandidateState.REJECTED,
            CandidateState.INCONCLUSIVE,
        }:
            raise SearchJournalIntegrityError(
                "terminal authorization does not contain a gate terminal state"
            )
        return authorization

    def _verify_feedback_disclosure(
        self,
        ref: ArtifactRef,
        *,
        round_index: int,
        candidate_ref: ArtifactRef,
        terminal_ref: ArtifactRef,
        authorization: TerminalTransitionAuthorization,
        champion_ref: ArtifactRef,
    ) -> AggregateFeedbackDisclosure:
        if ref.media_type != AGGREGATE_FEEDBACK_DISCLOSURE_MEDIA_TYPE:
            raise SearchJournalIntegrityError(
                "feedback disclosure ref declares the wrong media type"
            )
        disclosure = self.store.get_json(ref, AggregateFeedbackDisclosure)
        manifest = self.manifest
        expected = (
            self.controller_manifest_ref,
            manifest.search_run_ref,
            manifest.study_ref,
            manifest.experiment_ref,
            manifest.run_id,
            manifest.baseline_kind,
            manifest.search_seed,
            round_index,
            candidate_ref,
            terminal_ref,
            authorization.terminal_state,
            champion_ref,
        )
        actual = (
            disclosure.controller_manifest_ref,
            disclosure.search_run_ref,
            disclosure.study_ref,
            disclosure.experiment_ref,
            disclosure.run_id,
            disclosure.baseline_kind,
            disclosure.search_seed,
            disclosure.round_index,
            disclosure.candidate_ref,
            disclosure.terminal_authorization_ref,
            disclosure.terminal_state,
            disclosure.champion_harness_ref,
        )
        if actual != expected:
            raise SearchJournalIntegrityError("aggregate feedback disclosure is misbound")
        derived_summary = self._derive_feedback_summary(authorization)
        if disclosure.summary != derived_summary:
            raise SearchJournalIntegrityError(
                "aggregate feedback does not match the trusted gate decision"
            )
        return disclosure

    def _derive_feedback_summary(
        self, authorization: TerminalTransitionAuthorization
    ) -> AggregateFeedbackSummary:
        """Redact a verifier-bound GateDecision to a fixed aggregate schema."""

        try:
            decision = self.store.get_json(authorization.decision_ref, GateDecision)
        except Exception as exc:
            raise SearchJournalIntegrityError(
                f"trusted terminal decision could not be loaded: {exc}"
            ) from exc
        expected_state = {
            Decision.PROMOTE: CandidateState.PROMOTED,
            Decision.REJECT: CandidateState.REJECTED,
            Decision.INCONCLUSIVE: CandidateState.INCONCLUSIVE,
        }[decision.decision]
        if expected_state is not authorization.gate_terminal_state:
            raise SearchJournalIntegrityError(
                "gate decision does not match the terminal authorization"
            )
        authorized_decision = {
            CandidateState.PROMOTED: Decision.PROMOTE,
            CandidateState.REJECTED: Decision.REJECT,
            CandidateState.INCONCLUSIVE: Decision.INCONCLUSIVE,
        }[authorization.terminal_state]
        redacted_checks = tuple(
            AggregateGateCheckOutcome(name=check.name, outcome=check.outcome)
            for check in decision.checks
        )
        metrics = decision.comparison.metrics
        if metrics is None:
            return AggregateFeedbackSummary(
                decision=authorized_decision,
                gate_version=decision.gate_version,
                gate_config_sha256=decision.gate_config_sha256,
                checks=redacted_checks,
                n_valid_pairs=0,
                n_tasks=0,
                mean_score_delta=None,
                confidence_lower_bound=None,
                confidence_upper_bound=None,
                confidence_level=None,
                regression_rate=None,
                tokens_ratio=None,
                latency_ratio=None,
                tool_calls_ratio=None,
            )
        return AggregateFeedbackSummary(
            decision=authorized_decision,
            gate_version=decision.gate_version,
            gate_config_sha256=decision.gate_config_sha256,
            checks=redacted_checks,
            n_valid_pairs=metrics.n_valid_pairs,
            n_tasks=metrics.n_tasks,
            mean_score_delta=metrics.mean_delta,
            confidence_lower_bound=metrics.confidence_interval.lower,
            confidence_upper_bound=metrics.confidence_interval.upper,
            confidence_level=metrics.confidence_interval.confidence_level,
            regression_rate=metrics.regression_rate,
            tokens_ratio=metrics.tokens_ratio,
            latency_ratio=metrics.latency_ratio,
            tool_calls_ratio=metrics.tool_calls_ratio,
        )

    def _verify_selection_closure(
        self,
        ref: ArtifactRef,
        *,
        before_tail: ArtifactRef,
        champion_ref: ArtifactRef,
        champion_candidate_ref: ArtifactRef | None,
        stop_reasons: tuple[SearchStopReason, ...],
        completed_rounds: int,
        gate_nominations: int,
    ) -> SearchSelectionClosure:
        if ref.media_type != SEARCH_SELECTION_CLOSURE_MEDIA_TYPE:
            raise SearchJournalIntegrityError("selection closure ref declares the wrong media type")
        closure = self.store.get_json(ref, SearchSelectionClosure)
        manifest = self.manifest
        expected = (
            self.controller_manifest_ref,
            manifest.search_run_ref,
            manifest.study_ref,
            manifest.experiment_ref,
            manifest.run_id,
            manifest.baseline_kind,
            manifest.search_seed,
            before_tail,
            champion_ref,
            champion_candidate_ref,
            manifest.analysis_plan_ref,
            stop_reasons,
            completed_rounds,
            gate_nominations,
        )
        actual = (
            closure.controller_manifest_ref,
            closure.search_run_ref,
            closure.study_ref,
            closure.experiment_ref,
            closure.run_id,
            closure.baseline_kind,
            closure.search_seed,
            closure.search_tail_before_close_ref,
            closure.champion_harness_ref,
            closure.champion_candidate_ref,
            closure.analysis_plan_ref,
            closure.stop_reasons,
            closure.completed_rounds,
            closure.gate_nominations,
        )
        if actual != expected:
            raise SearchJournalIntegrityError("selection closure is misbound")
        return closure

    def _replay_semantics(
        self, events: tuple[tuple[ArtifactRef, SearchJournalEvent], ...]
    ) -> SearchRunSnapshot:
        manifest = self.manifest
        state: SearchState | None = None
        active_round: int | None = None
        stage: RoundStage | None = None
        completed_rounds = 0
        gate_nominations = 0
        rounds_since_promotion = 0
        consecutive_declines = 0
        champion = manifest.initial_champion_harness_ref
        champion_candidate: ArtifactRef | None = None
        candidate_refs: tuple[ArtifactRef, ...] = ()
        eligible_refs: tuple[ArtifactRef, ...] = ()
        nomination_ref: ArtifactRef | None = None
        nominated: ArtifactRef | None = None
        proposal_declined = False
        terminal_refs: list[ArtifactRef] = []
        feedback_refs: list[ArtifactRef] = []
        selection_closure_ref: ArtifactRef | None = None
        closed_stop_reasons: tuple[SearchStopReason, ...] = ()

        for index, (_event_ref, event) in enumerate(events):
            self._verify_linked_event_artifacts(event)
            if event.from_state is not state:
                expected_name = "ROOT" if state is None else state.name
                raise SearchJournalIntegrityError(
                    f"state discontinuity at sequence {event.sequence}: expected {expected_name}"
                )
            if event.champion_before_ref != champion:
                raise SearchJournalIntegrityError(
                    f"champion discontinuity at sequence {event.sequence}"
                )

            if event.kind is SearchEventKind.FROZEN:
                if index != 0:
                    raise SearchJournalIntegrityError("FROZEN must be the journal root")
            elif event.kind is SearchEventKind.SEARCH_STARTED:
                if active_round is not None or stage is not None:
                    raise SearchJournalIntegrityError("search started with an active round")
            elif event.kind is SearchEventKind.ROUND_OPENED:
                reasons = _stop_reasons(
                    manifest,
                    completed_rounds=completed_rounds,
                    gate_nominations=gate_nominations,
                    rounds_since_promotion=rounds_since_promotion,
                    consecutive_declines=consecutive_declines,
                )
                if reasons:
                    raise SearchJournalIntegrityError(
                        "round opened after a frozen stop criterion was met"
                    )
                if active_round is not None or stage is not None:
                    raise SearchJournalIntegrityError("round opened before prior round completed")
                if event.round_index != completed_rounds:
                    raise SearchJournalIntegrityError("round index is not contiguous from zero")
                active_round = completed_rounds
                stage = RoundStage.ROUND_OPEN
                candidate_refs = ()
                eligible_refs = ()
                nomination_ref = None
                nominated = None
                proposal_declined = False
            elif event.kind is SearchEventKind.EVIDENCE_RECORDED:
                self._require_round_join(event, active_round, stage)
                stage = RoundStage.EVIDENCE_READY
            elif event.kind is SearchEventKind.CANDIDATE_POOL_RECORDED:
                self._require_round_join(event, active_round, stage)
                candidate_refs = event.candidate_refs
                proposal_declined = event.proposal_declined
                stage = RoundStage.CANDIDATE_POOL_READY
            elif event.kind is SearchEventKind.SCREEN_RECORDED:
                self._require_round_join(event, active_round, stage)
                if not set(event.eligible_candidate_refs).issubset(candidate_refs):
                    raise SearchJournalIntegrityError(
                        "screened candidates are not a subset of the candidate pool"
                    )
                eligible_refs = event.eligible_candidate_refs
                stage = RoundStage.SCREENED
            elif event.kind is SearchEventKind.CANDIDATE_NOMINATED:
                self._require_round_join(event, active_round, stage)
                if event.nominated_candidate_ref not in eligible_refs:
                    raise SearchJournalIntegrityError(
                        "nominated candidate was not eligible after screening"
                    )
                if gate_nominations >= manifest.max_gate_nominations:
                    raise SearchJournalIntegrityError("gate nomination limit was exceeded")
                nominated = event.nominated_candidate_ref
                nomination_ref = event.nomination_ref
                gate_nominations += 1
                stage = RoundStage.NOMINATED
            elif event.kind is SearchEventKind.NO_CANDIDATE_RECORDED:
                self._require_round_join(event, active_round, stage)
                if eligible_refs:
                    raise SearchJournalIntegrityError(
                        "NO_CANDIDATE cannot discard screened eligible candidates"
                    )
                stage = RoundStage.NO_CANDIDATE
            elif event.kind is SearchEventKind.NOMINATED_ROUND_COMPLETED:
                self._require_round_join(event, active_round, stage)
                if nominated is None or event.nominated_candidate_ref != nominated:
                    raise SearchJournalIntegrityError(
                        "terminal event does not match the nominated candidate"
                    )
                terminal_ref = event.terminal_authorization_ref
                feedback_ref = event.feedback_disclosure_ref
                assert terminal_ref is not None and feedback_ref is not None
                authorization = self._verify_terminal_authorization(
                    terminal_ref,
                    candidate_ref=nominated,
                    champion_ref=champion,
                )
                next_champion = (
                    authorization.child_harness_ref
                    if authorization.terminal_state is CandidateState.PROMOTED
                    else champion
                )
                if event.champion_after_ref != next_champion:
                    raise SearchJournalIntegrityError(
                        "champion advancement does not match terminal authorization"
                    )
                self._verify_feedback_disclosure(
                    feedback_ref,
                    round_index=active_round,
                    candidate_ref=nominated,
                    terminal_ref=terminal_ref,
                    authorization=authorization,
                    champion_ref=next_champion,
                )
                terminal_refs.append(terminal_ref)
                feedback_refs.append(feedback_ref)
                completed_rounds += 1
                if authorization.terminal_state is CandidateState.PROMOTED:
                    champion = next_champion
                    champion_candidate = nominated
                    rounds_since_promotion = 0
                else:
                    rounds_since_promotion += 1
                consecutive_declines = 0
                active_round = None
                stage = None
                candidate_refs = ()
                eligible_refs = ()
                nomination_ref = None
                nominated = None
                proposal_declined = False
            elif event.kind is SearchEventKind.EMPTY_ROUND_COMPLETED:
                self._require_round_join(event, active_round, stage)
                if event.champion_after_ref != champion:
                    raise SearchJournalIntegrityError("empty round changed champion")
                completed_rounds += 1
                rounds_since_promotion += 1
                consecutive_declines = consecutive_declines + 1 if proposal_declined else 0
                active_round = None
                stage = None
                candidate_refs = ()
                eligible_refs = ()
                nomination_ref = None
                nominated = None
                proposal_declined = False
            elif event.kind is SearchEventKind.SELECTION_CLOSED:
                if active_round is not None or stage is not None:
                    raise SearchJournalIntegrityError("selection closed during an active round")
                reasons = _stop_reasons(
                    manifest,
                    completed_rounds=completed_rounds,
                    gate_nominations=gate_nominations,
                    rounds_since_promotion=rounds_since_promotion,
                    consecutive_declines=consecutive_declines,
                )
                if event.stop_reasons != reasons or not reasons:
                    raise SearchJournalIntegrityError(
                        "selection closure does not match replay-derived stop reasons"
                    )
                closure_ref = event.selection_closure_ref
                assert closure_ref is not None and event.previous_tail_ref is not None
                self._verify_selection_closure(
                    closure_ref,
                    before_tail=event.previous_tail_ref,
                    champion_ref=champion,
                    champion_candidate_ref=champion_candidate,
                    stop_reasons=reasons,
                    completed_rounds=completed_rounds,
                    gate_nominations=gate_nominations,
                )
                selection_closure_ref = closure_ref
                closed_stop_reasons = reasons
            elif event.kind is SearchEventKind.INVALIDATED:
                if event.invalidation_ref is None:  # model validation also enforces this
                    raise SearchJournalIntegrityError("invalidation evidence is missing")
                if event.from_stage is not stage or event.round_index != active_round:
                    raise SearchJournalIntegrityError(
                        "invalidation does not bind the active round stage"
                    )
                _require_json_ref(event.invalidation_ref, field_name="invalidation_ref")
                active_round = None
                stage = None
                candidate_refs = ()
                eligible_refs = ()
                nomination_ref = None
                nominated = None
                proposal_declined = False
            else:  # pragma: no cover - exhaustive enum dispatch
                raise SearchJournalIntegrityError(f"unsupported event kind: {event.kind}")

            state = event.to_state
            if (
                event.kind is not SearchEventKind.NOMINATED_ROUND_COMPLETED
                and event.champion_after_ref != champion
            ):
                raise SearchJournalIntegrityError(
                    f"event changed champion at sequence {event.sequence}"
                )

        assert state is not None
        tail_ref = events[-1][0]
        available_stop_reasons = _stop_reasons(
            manifest,
            completed_rounds=completed_rounds,
            gate_nominations=gate_nominations,
            rounds_since_promotion=rounds_since_promotion,
            consecutive_declines=consecutive_declines,
        )
        return SearchRunSnapshot(
            controller_manifest_ref=self.controller_manifest_ref,
            search_run_ref=manifest.search_run_ref,
            study_ref=manifest.study_ref,
            experiment_ref=manifest.experiment_ref,
            run_id=manifest.run_id,
            baseline_kind=manifest.baseline_kind,
            search_seed=manifest.search_seed,
            tail_ref=tail_ref,
            sequence=events[-1][1].sequence,
            event_refs=tuple(ref for ref, _ in events),
            state=state,
            active_round_index=active_round,
            round_stage=stage,
            completed_rounds=completed_rounds,
            gate_nominations=gate_nominations,
            rounds_since_promotion=rounds_since_promotion,
            consecutive_declines=consecutive_declines,
            champion_harness_ref=champion,
            champion_candidate_ref=champion_candidate,
            candidate_refs=candidate_refs,
            eligible_candidate_refs=eligible_refs,
            nomination_ref=nomination_ref,
            nominated_candidate_ref=nominated,
            current_proposal_declined=proposal_declined,
            terminal_authorization_refs=tuple(terminal_refs),
            feedback_disclosure_refs=tuple(feedback_refs),
            last_feedback_disclosure_ref=(feedback_refs[-1] if feedback_refs else None),
            selection_closure_ref=selection_closure_ref,
            stop_reasons=(closed_stop_reasons or available_stop_reasons),
            proposal_api_open=(
                state is SearchState.SEARCHING
                and not available_stop_reasons
                and active_round is None
            ),
        )

    def _verify_linked_event_artifacts(self, event: SearchJournalEvent) -> None:
        refs = (
            *event.candidate_refs,
            *event.eligible_candidate_refs,
            event.evidence_packet_ref,
            event.candidate_pool_ref,
            event.screen_report_ref,
            event.nomination_ref,
            event.nominated_candidate_ref,
            event.invalidation_ref,
        )
        for ref in refs:
            if ref is None:
                continue
            try:
                self.store.get_json(ref)
            except Exception as exc:
                raise SearchJournalIntegrityError(
                    f"event artifact {ref.sha256} could not be verified: {exc}"
                ) from exc

    @staticmethod
    def _require_round_join(
        event: SearchJournalEvent,
        active_round: int | None,
        stage: RoundStage | None,
    ) -> None:
        if active_round is None or event.round_index != active_round:
            raise SearchJournalIntegrityError("event belongs to a foreign or inactive round")
        if event.from_stage is not stage:
            expected = "NONE" if stage is None else stage.name
            raise SearchJournalIntegrityError(
                f"round-stage discontinuity: expected {expected}, got {event.from_stage}"
            )


class SearchController:
    """Single-writer semantic owner for one arm and one independent search run."""

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        controller_manifest_ref: ArtifactRef,
        event_service: TrustedSearchEventService,
        terminal_authorization_verifier: TerminalAuthorizationVerifier | None = None,
        tail_ref: ArtifactRef | None = None,
    ) -> None:
        if type(event_service) is not TrustedSearchEventService:
            raise TypeError("event_service must be an exact TrustedSearchEventService")
        self.store = store
        self.controller_manifest_ref = ArtifactRef.model_validate(controller_manifest_ref)
        self.event_service = event_service
        self.journal = SearchJournal(
            store,
            controller_manifest_ref=self.controller_manifest_ref,
            event_verifier=event_service.verification_capability,
            terminal_authorization_verifier=terminal_authorization_verifier,
        )
        self.manifest = self.journal.manifest
        self._tail_ref: ArtifactRef | None = None
        self._snapshot: SearchRunSnapshot | None = None
        if tail_ref is not None:
            self._snapshot = self.journal.replay(ArtifactRef.model_validate(tail_ref))
            self._tail_ref = self._snapshot.tail_ref

    @property
    def tail_ref(self) -> ArtifactRef | None:
        return self._tail_ref

    @property
    def snapshot(self) -> SearchRunSnapshot | None:
        return self._snapshot

    @property
    def state(self) -> SearchState | None:
        return None if self._snapshot is None else self._snapshot.state

    def freeze(self) -> ArtifactRef:
        """Publish the attested FROZEN root exactly once."""

        if self._tail_ref is not None:
            raise StaleSearchTailError("search controller is already frozen")
        return self._append(
            kind=SearchEventKind.FROZEN,
            previous_tail_ref=None,
            reason="frozen search-controller manifest accepted",
        )

    def start_search(self, *, previous_tail_ref: ArtifactRef) -> ArtifactRef:
        snapshot = self._require_tail(previous_tail_ref)
        self._require_state(snapshot, SearchState.FROZEN)
        return self._append(
            kind=SearchEventKind.SEARCH_STARTED,
            previous_tail_ref=previous_tail_ref,
            reason="bounded search started",
        )

    def open_round(self, *, previous_tail_ref: ArtifactRef) -> ArtifactRef:
        snapshot = self._require_tail(previous_tail_ref)
        self._require_idle_search(snapshot)
        reasons = self._available_stop_reasons(snapshot)
        if reasons:
            joined = ", ".join(reason.value for reason in reasons)
            raise SearchStopError(f"cannot open a round after stop criterion: {joined}")
        return self._append(
            kind=SearchEventKind.ROUND_OPENED,
            previous_tail_ref=previous_tail_ref,
            round_index=snapshot.completed_rounds,
            reason="search round opened",
        )

    def record_evidence(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        evidence_packet_ref: ArtifactRef,
    ) -> ArtifactRef:
        snapshot = self._require_active_stage(previous_tail_ref, RoundStage.ROUND_OPEN)
        validated_ref = self._validated_json_ref(
            evidence_packet_ref, field_name="evidence_packet_ref"
        )
        return self._append(
            kind=SearchEventKind.EVIDENCE_RECORDED,
            previous_tail_ref=previous_tail_ref,
            round_index=snapshot.active_round_index,
            evidence_packet_ref=validated_ref,
            reason="exploration evidence packet frozen",
        )

    def record_candidate_pool(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        candidate_pool_ref: ArtifactRef,
        candidate_refs: tuple[ArtifactRef, ...] = (),
        declined: bool = False,
    ) -> ArtifactRef:
        snapshot = self._require_active_stage(previous_tail_ref, RoundStage.EVIDENCE_READY)
        pool_ref = self._validated_json_ref(candidate_pool_ref, field_name="candidate_pool_ref")
        validated_candidates = tuple(
            self._validated_json_ref(ref, field_name="candidate_ref") for ref in candidate_refs
        )
        if declined and validated_candidates:
            raise SearchControllerError("a declined proposal cannot publish candidates")
        return self._append(
            kind=SearchEventKind.CANDIDATE_POOL_RECORDED,
            previous_tail_ref=previous_tail_ref,
            round_index=snapshot.active_round_index,
            candidate_pool_ref=pool_ref,
            candidate_refs=validated_candidates,
            proposal_declined=declined,
            reason=("proposer declined mutation" if declined else "candidate pool frozen"),
        )

    def record_screen(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        screen_report_ref: ArtifactRef,
        eligible_candidate_refs: tuple[ArtifactRef, ...] = (),
    ) -> ArtifactRef:
        snapshot = self._require_active_stage(previous_tail_ref, RoundStage.CANDIDATE_POOL_READY)
        report_ref = self._validated_json_ref(screen_report_ref, field_name="screen_report_ref")
        eligible = tuple(
            self._validated_json_ref(ref, field_name="eligible_candidate_ref")
            for ref in eligible_candidate_refs
        )
        if not set(eligible).issubset(snapshot.candidate_refs):
            raise SearchControllerError(
                "eligible candidates must be a subset of the frozen candidate pool"
            )
        return self._append(
            kind=SearchEventKind.SCREEN_RECORDED,
            previous_tail_ref=previous_tail_ref,
            round_index=snapshot.active_round_index,
            screen_report_ref=report_ref,
            eligible_candidate_refs=eligible,
            reason="candidate screen completed",
        )

    def nominate(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        nomination_ref: ArtifactRef,
        candidate_ref: ArtifactRef,
    ) -> ArtifactRef:
        snapshot = self._require_active_stage(previous_tail_ref, RoundStage.SCREENED)
        validated_nomination = self._validated_json_ref(nomination_ref, field_name="nomination_ref")
        validated_candidate = self._validated_json_ref(candidate_ref, field_name="candidate_ref")
        if validated_candidate not in snapshot.eligible_candidate_refs:
            raise SearchControllerError("candidate was not eligible after screening")
        if snapshot.gate_nominations >= self.manifest.max_gate_nominations:
            raise SearchStopError("maximum gate nominations already reached")
        return self._append(
            kind=SearchEventKind.CANDIDATE_NOMINATED,
            previous_tail_ref=previous_tail_ref,
            round_index=snapshot.active_round_index,
            nomination_ref=validated_nomination,
            nominated_candidate_ref=validated_candidate,
            reason="one screened candidate nominated for gate evaluation",
        )

    def record_no_candidate(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        reason: str,
    ) -> ArtifactRef:
        snapshot = self._require_active_stage(previous_tail_ref, RoundStage.SCREENED)
        if snapshot.eligible_candidate_refs:
            raise SearchControllerError(
                "cannot record NO_CANDIDATE while screened candidates remain eligible"
            )
        return self._append(
            kind=SearchEventKind.NO_CANDIDATE_RECORDED,
            previous_tail_ref=previous_tail_ref,
            round_index=snapshot.active_round_index,
            reason=self._normalize_reason(reason),
        )

    def complete_nominated_round(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        terminal_authorization_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Resolve one nomination through an authenticated ExperimentController result."""

        snapshot = self._require_active_stage(previous_tail_ref, RoundStage.NOMINATED)
        candidate_ref = snapshot.nominated_candidate_ref
        round_index = snapshot.active_round_index
        assert candidate_ref is not None and round_index is not None
        terminal_ref = ArtifactRef.model_validate(terminal_authorization_ref)
        authorization = self.journal._verify_terminal_authorization(
            terminal_ref,
            candidate_ref=candidate_ref,
            champion_ref=snapshot.champion_harness_ref,
        )
        next_champion = (
            authorization.child_harness_ref
            if authorization.terminal_state is CandidateState.PROMOTED
            else snapshot.champion_harness_ref
        )
        summary = self.journal._derive_feedback_summary(authorization)
        disclosure = AggregateFeedbackDisclosure(
            controller_manifest_ref=self.controller_manifest_ref,
            search_run_ref=self.manifest.search_run_ref,
            study_ref=self.manifest.study_ref,
            experiment_ref=self.manifest.experiment_ref,
            run_id=self.manifest.run_id,
            baseline_kind=self.manifest.baseline_kind,
            search_seed=self.manifest.search_seed,
            round_index=round_index,
            candidate_ref=candidate_ref,
            terminal_authorization_ref=terminal_ref,
            terminal_state=authorization.terminal_state,
            champion_harness_ref=next_champion,
            gate_version=summary.gate_version,
            gate_config_sha256=summary.gate_config_sha256,
            summary=summary,
        )
        disclosure_ref = self.store.put_json(
            disclosure, media_type=AGGREGATE_FEEDBACK_DISCLOSURE_MEDIA_TYPE
        )
        return self._append(
            kind=SearchEventKind.NOMINATED_ROUND_COMPLETED,
            previous_tail_ref=previous_tail_ref,
            round_index=round_index,
            nominated_candidate_ref=candidate_ref,
            terminal_authorization_ref=terminal_ref,
            feedback_disclosure_ref=disclosure_ref,
            champion_after_ref=next_champion,
            reason=f"nominated candidate resolved as {authorization.terminal_state.value}",
        )

    def complete_no_candidate_round(self, *, previous_tail_ref: ArtifactRef) -> ArtifactRef:
        snapshot = self._require_active_stage(previous_tail_ref, RoundStage.NO_CANDIDATE)
        return self._append(
            kind=SearchEventKind.EMPTY_ROUND_COMPLETED,
            previous_tail_ref=previous_tail_ref,
            round_index=snapshot.active_round_index,
            reason="round completed without a gate nomination",
        )

    def close_selection(self, *, previous_tail_ref: ArtifactRef) -> ArtifactRef:
        """Close only for replay-derived limits; analysis plan is never a parameter."""

        snapshot = self._require_tail(previous_tail_ref)
        self._require_idle_search(snapshot)
        reasons = self._available_stop_reasons(snapshot)
        if not reasons:
            raise SearchStopError("selection cannot close before a frozen stop criterion is met")
        closure = SearchSelectionClosure(
            controller_manifest_ref=self.controller_manifest_ref,
            search_run_ref=self.manifest.search_run_ref,
            study_ref=self.manifest.study_ref,
            experiment_ref=self.manifest.experiment_ref,
            run_id=self.manifest.run_id,
            baseline_kind=self.manifest.baseline_kind,
            search_seed=self.manifest.search_seed,
            search_tail_before_close_ref=snapshot.tail_ref,
            champion_harness_ref=snapshot.champion_harness_ref,
            champion_candidate_ref=snapshot.champion_candidate_ref,
            analysis_plan_ref=self.manifest.analysis_plan_ref,
            stop_reasons=reasons,
            completed_rounds=snapshot.completed_rounds,
            gate_nominations=snapshot.gate_nominations,
        )
        closure_ref = self.store.put_json(closure, media_type=SEARCH_SELECTION_CLOSURE_MEDIA_TYPE)
        return self._append(
            kind=SearchEventKind.SELECTION_CLOSED,
            previous_tail_ref=previous_tail_ref,
            selection_closure_ref=closure_ref,
            stop_reasons=reasons,
            reason="bounded search selection permanently closed",
        )

    def invalidate(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        invalidation_ref: ArtifactRef,
        reason: str,
    ) -> ArtifactRef:
        snapshot = self._require_tail(previous_tail_ref)
        if snapshot.state is SearchState.INVALIDATED:
            raise SearchControllerError("search run is already invalidated")
        validated_ref = self._validated_json_ref(invalidation_ref, field_name="invalidation_ref")
        return self._append(
            kind=SearchEventKind.INVALIDATED,
            previous_tail_ref=previous_tail_ref,
            round_index=snapshot.active_round_index,
            from_stage=snapshot.round_stage,
            invalidation_ref=validated_ref,
            reason=self._normalize_reason(reason),
        )

    def verify_selection_closure(self, tail_ref: ArtifactRef) -> SearchRunSnapshot:
        """Authenticate the current selected tail for a study-level barrier."""

        validated = ArtifactRef.model_validate(tail_ref)
        if self._tail_ref != validated:
            raise StaleSearchTailError("selection closure tail is stale or foreign")
        snapshot = self.journal.replay(validated)
        if snapshot.state is not SearchState.SELECTION_CLOSED:
            raise SearchControllerError("tail is not in SELECTION_CLOSED state")
        if snapshot.selection_closure_ref is None:  # pragma: no cover - replay enforces
            raise SearchJournalIntegrityError("selection closure artifact is missing")
        self._snapshot = snapshot
        return snapshot

    def _append(
        self,
        *,
        kind: SearchEventKind,
        previous_tail_ref: ArtifactRef | None,
        reason: str,
        round_index: int | None = None,
        from_stage: RoundStage | None = None,
        champion_after_ref: ArtifactRef | None = None,
        evidence_packet_ref: ArtifactRef | None = None,
        candidate_pool_ref: ArtifactRef | None = None,
        candidate_refs: tuple[ArtifactRef, ...] = (),
        proposal_declined: bool = False,
        screen_report_ref: ArtifactRef | None = None,
        eligible_candidate_refs: tuple[ArtifactRef, ...] = (),
        nomination_ref: ArtifactRef | None = None,
        nominated_candidate_ref: ArtifactRef | None = None,
        terminal_authorization_ref: ArtifactRef | None = None,
        feedback_disclosure_ref: ArtifactRef | None = None,
        selection_closure_ref: ArtifactRef | None = None,
        invalidation_ref: ArtifactRef | None = None,
        stop_reasons: tuple[SearchStopReason, ...] = (),
    ) -> ArtifactRef:
        snapshot = self._snapshot
        sequence = 0 if snapshot is None else snapshot.sequence + 1
        champion_before = (
            self.manifest.initial_champion_harness_ref
            if snapshot is None
            else snapshot.champion_harness_ref
        )
        if kind is SearchEventKind.INVALIDATED:
            from_state = None if snapshot is None else snapshot.state
            to_state = SearchState.INVALIDATED
            to_stage = None
        else:
            from_state, to_state, expected_from_stage, to_stage = _EVENT_SHAPES[kind]
            if from_stage is None:
                from_stage = expected_from_stage
        content = SearchJournalEventContent(
            controller_manifest_ref=self.controller_manifest_ref,
            search_run_ref=self.manifest.search_run_ref,
            study_ref=self.manifest.study_ref,
            experiment_ref=self.manifest.experiment_ref,
            run_id=self.manifest.run_id,
            baseline_kind=self.manifest.baseline_kind,
            search_seed=self.manifest.search_seed,
            sequence=sequence,
            previous_tail_ref=previous_tail_ref,
            kind=kind,
            from_state=from_state,
            to_state=to_state,
            round_index=round_index,
            from_stage=from_stage,
            to_stage=to_stage,
            champion_before_ref=champion_before,
            champion_after_ref=champion_after_ref or champion_before,
            evidence_packet_ref=evidence_packet_ref,
            candidate_pool_ref=candidate_pool_ref,
            candidate_refs=candidate_refs,
            proposal_declined=proposal_declined,
            screen_report_ref=screen_report_ref,
            eligible_candidate_refs=eligible_candidate_refs,
            nomination_ref=nomination_ref,
            nominated_candidate_ref=nominated_candidate_ref,
            terminal_authorization_ref=terminal_authorization_ref,
            feedback_disclosure_ref=feedback_disclosure_ref,
            selection_closure_ref=selection_closure_ref,
            invalidation_ref=invalidation_ref,
            stop_reasons=stop_reasons,
            reason=reason,
        )
        event = TrustedSearchEventService._issue(self.event_service, content)
        tail_ref = self.store.put_json(event, media_type=SEARCH_JOURNAL_EVENT_MEDIA_TYPE)
        replayed = self.journal.replay(tail_ref)
        self._tail_ref = tail_ref
        self._snapshot = replayed
        return tail_ref

    def _require_tail(self, previous_tail_ref: ArtifactRef) -> SearchRunSnapshot:
        validated = ArtifactRef.model_validate(previous_tail_ref)
        if validated.media_type != SEARCH_JOURNAL_EVENT_MEDIA_TYPE:
            raise StaleSearchTailError("previous tail declares the wrong media type")
        if self._tail_ref is None or validated != self._tail_ref:
            raise StaleSearchTailError("previous search tail is stale or foreign")
        snapshot = self.journal.replay(validated)
        self._snapshot = snapshot
        return snapshot

    @staticmethod
    def _require_state(snapshot: SearchRunSnapshot, expected: SearchState) -> None:
        if snapshot.state is not expected:
            raise SearchControllerError(
                f"expected search state {expected.name}, got {snapshot.state.name}"
            )

    def _require_idle_search(self, snapshot: SearchRunSnapshot) -> None:
        self._require_state(snapshot, SearchState.SEARCHING)
        if snapshot.active_round_index is not None or snapshot.round_stage is not None:
            raise SearchControllerError("operation requires no active search round")

    def _require_active_stage(
        self, previous_tail_ref: ArtifactRef, expected: RoundStage
    ) -> SearchRunSnapshot:
        snapshot = self._require_tail(previous_tail_ref)
        self._require_state(snapshot, SearchState.SEARCHING)
        if snapshot.active_round_index is None or snapshot.round_stage is not expected:
            actual = "NONE" if snapshot.round_stage is None else snapshot.round_stage.name
            raise SearchControllerError(
                f"expected active round stage {expected.name}, got {actual}"
            )
        return snapshot

    def _available_stop_reasons(self, snapshot: SearchRunSnapshot) -> tuple[SearchStopReason, ...]:
        return _stop_reasons(
            self.manifest,
            completed_rounds=snapshot.completed_rounds,
            gate_nominations=snapshot.gate_nominations,
            rounds_since_promotion=snapshot.rounds_since_promotion,
            consecutive_declines=snapshot.consecutive_declines,
        )

    def _validated_json_ref(self, ref: ArtifactRef, *, field_name: str) -> ArtifactRef:
        validated = ArtifactRef.model_validate(ref)
        _require_json_ref(validated, field_name=field_name)
        try:
            self.store.get_json(validated)
        except Exception as exc:
            raise SearchControllerError(
                f"{field_name} could not be verified in the artifact store: {exc}"
            ) from exc
        return validated

    @staticmethod
    def _normalize_reason(reason: str) -> str:
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        normalized = reason.strip()
        if not normalized:
            raise ValueError("reason must not be empty")
        return normalized


__all__ = [
    "AGGREGATE_FEEDBACK_DISCLOSURE_MEDIA_TYPE",
    "SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE",
    "SEARCH_JOURNAL_EVENT_MEDIA_TYPE",
    "SEARCH_SELECTION_CLOSURE_MEDIA_TYPE",
    "AggregateFeedbackDisclosure",
    "AggregateFeedbackSummary",
    "AggregateGateCheckOutcome",
    "RoundStage",
    "SearchController",
    "SearchControllerError",
    "SearchControllerManifest",
    "SearchEventAttestationError",
    "SearchEventKind",
    "SearchEventVerificationCapability",
    "SearchJournal",
    "SearchJournalCycleError",
    "SearchJournalEvent",
    "SearchJournalEventContent",
    "SearchJournalIntegrityError",
    "SearchRunSnapshot",
    "SearchSelectionClosure",
    "SearchState",
    "SearchStopError",
    "SearchStopReason",
    "StaleSearchTailError",
    "TerminalAuthorizationVerifier",
    "TrustedSearchEventService",
]
