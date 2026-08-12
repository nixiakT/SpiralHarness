"""Controller-owned authorization for one preregistered skill-probe execution."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Sequence
from threading import Lock
from typing import Literal

from pydantic import model_validator

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
)
from spiral_harness.core.lifecycle import CandidateLifecycleEvent, CandidateState
from spiral_harness.core.models import ArtifactRef, ImmutableModel, Sha256
from spiral_harness.storage.journal import JOURNAL_ENTRY_MEDIA_TYPE
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.skill_plan import SKILL_MECHANISM_PLAN_MEDIA_TYPE

from .admission import ADMISSION_REPORT_MEDIA_TYPE, CandidateAdmissionService
from .skill_probes import (
    SkillProbePreregistrationError,
    replay_probe_preregistration_refs,
)

SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-probe-execution-authorization.v1+json"
)
_MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.matched-skill-probe-closure.v1+json"
)


class SkillProbeExecutionAuthorizationError(RuntimeError):
    """Raised when a skill-probe execution is not authorized by the live controller."""


class _ConsumedSkillProbeExecutionAuthorizationError(
    SkillProbeExecutionAuthorizationError,
):
    """Signal that ``begin`` failed only after irreversibly consuming its grant."""


class SkillProbeExecutionAuthorization(ImmutableModel):
    """Bind one process-local execution grant to an exact probe lifecycle head."""

    schema_version: Literal["1"] = "1"
    execution_nonce: Sha256
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    plan_ref: ArtifactRef
    running_probes_tail_ref: ArtifactRef

    @model_validator(mode="after")
    def references_have_exact_media_types(self) -> SkillProbeExecutionAuthorization:
        expected = (
            ("experiment_ref", EXPERIMENT_MANIFEST_MEDIA_TYPE),
            ("protocol_ref", PROTOCOL_MANIFEST_MEDIA_TYPE),
            ("candidate_ref", CANDIDATE_MANIFEST_MEDIA_TYPE),
            ("plan_ref", SKILL_MECHANISM_PLAN_MEDIA_TYPE),
            ("running_probes_tail_ref", JOURNAL_ENTRY_MEDIA_TYPE),
        )
        for field_name, media_type in expected:
            if getattr(self, field_name).media_type != media_type:
                raise ValueError(f"{field_name} must declare {media_type!r}")
        return self


def _ref_key(ref: ArtifactRef) -> tuple[str, int, str]:
    return ref.sha256, ref.size, ref.media_type


def _load_exact_authorization(
    repository: ArtifactRepository,
    ref: ArtifactRef,
) -> SkillProbeExecutionAuthorization:
    try:
        checked_ref = ArtifactRef.model_validate(ref, strict=True)
    except Exception as exc:
        raise SkillProbeExecutionAuthorizationError(
            "skill-probe execution authorization reference is malformed"
        ) from exc
    if checked_ref.media_type != SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE:
        raise SkillProbeExecutionAuthorizationError(
            "skill-probe execution authorization declares the wrong media type"
        )
    try:
        payload = repository.get_bytes(checked_ref)
        authorization = repository.get_json(
            checked_ref,
            SkillProbeExecutionAuthorization,
        )
        checked = SkillProbeExecutionAuthorization.model_validate(
            authorization,
            strict=True,
        )
        canonical = canonical_json_bytes(checked)
    except Exception as exc:
        raise SkillProbeExecutionAuthorizationError(
            "skill-probe execution authorization cannot be loaded exactly"
        ) from exc
    if (
        payload != canonical
        or len(payload) != checked_ref.size
        or sha256_bytes(payload) != checked_ref.sha256
    ):
        raise SkillProbeExecutionAuthorizationError(
            "skill-probe execution authorization is not canonical under its reference"
        )
    return checked


def probe_plan_ref_from_history(
    events: Sequence[CandidateLifecycleEvent],
) -> ArtifactRef:
    """Extract the unique plan only from a canonical three-stage probe history."""

    expected_states = (
        CandidateState.REGISTERED,
        CandidateState.VALID,
        CandidateState.RUNNING_PROBES,
    )
    if tuple(event.to_state for event in events) != expected_states:
        raise SkillProbeExecutionAuthorizationError(
            "skill-probe authorization requires canonical three-event history"
        )
    if events[0].evidence_refs:
        raise SkillProbeExecutionAuthorizationError(
            "registered skill-probe history must not contain evidence"
        )
    if len(events[1].evidence_refs) != 1:
        raise SkillProbeExecutionAuthorizationError(
            "valid skill-probe history requires exactly one admission report"
        )
    if events[1].evidence_refs[0].media_type != ADMISSION_REPORT_MEDIA_TYPE:
        raise SkillProbeExecutionAuthorizationError(
            "valid skill-probe history binds the wrong admission report type"
        )
    if len(events[2].evidence_refs) != 1:
        raise SkillProbeExecutionAuthorizationError(
            "running-probes history requires exactly one skill mechanism plan"
        )
    plan_ref = events[2].evidence_refs[0]
    if plan_ref.media_type != SKILL_MECHANISM_PLAN_MEDIA_TYPE:
        raise SkillProbeExecutionAuthorizationError(
            "running-probes history binds the wrong skill mechanism plan type"
        )
    return plan_ref


def verify_skill_probe_execution_authorization_history(
    repository: ArtifactRepository,
    *,
    authorization: SkillProbeExecutionAuthorization,
    events: Sequence[CandidateLifecycleEvent],
) -> None:
    """Replay admission and preregistration for an already current controller head."""

    if any(event.candidate_ref != authorization.candidate_ref for event in events):
        raise SkillProbeExecutionAuthorizationError(
            "skill-probe authorization history belongs to another candidate"
        )
    plan_ref = probe_plan_ref_from_history(events)
    if plan_ref != authorization.plan_ref:
        raise SkillProbeExecutionAuthorizationError(
            "skill-probe authorization does not bind the lifecycle plan"
        )
    admission_ref = events[1].evidence_refs[0]
    try:
        CandidateAdmissionService(repository).verify_report(
            candidate_ref=authorization.candidate_ref,
            experiment_ref=authorization.experiment_ref,
            report_ref=admission_ref,
        )
        replay_probe_preregistration_refs(
            repository,
            experiment_ref=authorization.experiment_ref,
            protocol_ref=authorization.protocol_ref,
            candidate_ref=authorization.candidate_ref,
            evidence_refs=events[2].evidence_refs,
        )
    except Exception as exc:
        label = (
            "skill-probe admission replay failed"
            if not isinstance(exc, SkillProbePreregistrationError)
            else "skill-probe preregistration replay failed"
        )
        raise SkillProbeExecutionAuthorizationError(f"{label}: {exc}") from exc


class _SkillProbeExecutionAuthorizationState:
    """Shared mutable state that is reachable only through split capabilities."""

    __slots__ = (
        "__closures",
        "__issued",
        "__lock",
        "__repository",
        "__started",
        "__verify_current",
    )

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        verify_current: Callable[[SkillProbeExecutionAuthorization], None],
    ) -> None:
        if not isinstance(repository, ArtifactRepository):
            raise TypeError("repository must implement ArtifactRepository")
        if not callable(verify_current):
            raise TypeError("verify_current must be callable")
        self.__repository = repository
        self.__verify_current = verify_current
        self.__issued: dict[
            tuple[str, int, str],
            SkillProbeExecutionAuthorization,
        ] = {}
        self.__started: set[tuple[str, int, str]] = set()
        self.__closures: dict[tuple[str, int, str], ArtifactRef] = {}
        self.__lock = Lock()

    @property
    def repository(self) -> ArtifactRepository:
        return self.__repository

    def register_issued(
        self,
        ref: ArtifactRef,
        authorization: SkillProbeExecutionAuthorization,
    ) -> None:
        key = _ref_key(ref)
        with self.__lock:
            if key in self.__issued:
                raise SkillProbeExecutionAuthorizationError(
                    "skill-probe execution authorization was issued more than once"
                )
            self.__issued[key] = authorization

    def issued_authorization(
        self,
        ref: ArtifactRef,
    ) -> tuple[ArtifactRef, SkillProbeExecutionAuthorization]:
        try:
            checked_ref = ArtifactRef.model_validate(ref, strict=True)
        except Exception as exc:
            raise SkillProbeExecutionAuthorizationError(
                "skill-probe execution authorization reference is malformed"
            ) from exc
        with self.__lock:
            issued = self.__issued.get(_ref_key(checked_ref))
        if issued is None:
            raise SkillProbeExecutionAuthorizationError(
                "skill-probe execution authorization was not issued by this capability"
            )
        return checked_ref, issued

    def verify_issued(
        self,
        checked_ref: ArtifactRef,
        issued: SkillProbeExecutionAuthorization,
    ) -> SkillProbeExecutionAuthorization:
        persisted = _load_exact_authorization(self.__repository, checked_ref)
        if persisted != issued:
            raise SkillProbeExecutionAuthorizationError(
                "persisted skill-probe authorization differs from the issued grant"
            )
        try:
            self.__verify_current(persisted)
        except SkillProbeExecutionAuthorizationError:
            raise
        except Exception as exc:
            raise SkillProbeExecutionAuthorizationError(
                f"skill-probe authorization is not current: {exc}"
            ) from exc
        return persisted

    def verify_current_authorization(
        self,
        authorization: SkillProbeExecutionAuthorization,
    ) -> None:
        try:
            self.__verify_current(authorization)
        except SkillProbeExecutionAuthorizationError:
            raise
        except Exception as exc:
            raise SkillProbeExecutionAuthorizationError(
                f"skill-probe authorization is not current: {exc}"
            ) from exc

    def begin(
        self,
        ref: ArtifactRef,
    ) -> SkillProbeExecutionAuthorization:
        checked_ref, issued = self.issued_authorization(ref)
        key = _ref_key(checked_ref)
        with self.__lock:
            if key in self.__started:
                raise SkillProbeExecutionAuthorizationError(
                    "skill-probe execution authorization was already started"
                )
            # Starting precedes repository/current-head replay intentionally: a
            # failed attempt must burn the execution right rather than enable an
            # adaptive rerun.
            self.__started.add(key)
        try:
            return self.verify_issued(checked_ref, issued)
        except Exception as exc:
            raise _ConsumedSkillProbeExecutionAuthorizationError(
                "skill-probe authorization verification failed after execution started"
            ) from exc

    def register_closure(
        self,
        authorization_ref: ArtifactRef,
        closure_ref: ArtifactRef,
    ) -> SkillProbeExecutionAuthorization:
        checked_ref, issued = self.issued_authorization(authorization_ref)
        checked_closure_ref = _checked_closure_ref(closure_ref)
        key = _ref_key(checked_ref)
        with self.__lock:
            if key not in self.__started:
                raise SkillProbeExecutionAuthorizationError(
                    "skill-probe execution authorization was not started"
                )
            if key in self.__closures:
                raise SkillProbeExecutionAuthorizationError(
                    "skill-probe execution closure was already registered"
                )
            # Bind first and verify second.  A concurrent lifecycle advance can
            # invalidate this closure, but it can never reopen the grant or let
            # a caller substitute a second result.
            self.__closures[key] = checked_closure_ref
        return self.verify_issued(checked_ref, issued)

    def verify_registered_closure(
        self,
        authorization_ref: ArtifactRef,
        closure_ref: ArtifactRef,
    ) -> SkillProbeExecutionAuthorization:
        checked_ref, issued = self.issued_authorization(authorization_ref)
        checked_closure_ref = _checked_closure_ref(closure_ref)
        key = _ref_key(checked_ref)
        with self.__lock:
            registered = self.__closures.get(key)
        if registered is None:
            raise SkillProbeExecutionAuthorizationError(
                "skill-probe execution closure was not registered"
            )
        if registered != checked_closure_ref:
            raise SkillProbeExecutionAuthorizationError(
                "skill-probe execution authorization binds another closure"
            )
        return self.verify_issued(checked_ref, issued)


def _checked_closure_ref(ref: ArtifactRef) -> ArtifactRef:
    try:
        checked = ArtifactRef.model_validate(ref, strict=True)
    except Exception as exc:
        raise SkillProbeExecutionAuthorizationError(
            "skill-probe execution closure reference is malformed"
        ) from exc
    if checked.media_type != _MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE:
        raise SkillProbeExecutionAuthorizationError(
            "skill-probe execution closure declares the wrong media type"
        )
    return checked


def _require_trusted_state(
    owner: object,
    *,
    attribute: str,
    label: str,
) -> _SkillProbeExecutionAuthorizationState:
    """Reject exact-type shells that were not composed by the trusted service."""

    try:
        state = object.__getattribute__(owner, attribute)
    except AttributeError as exc:
        raise SkillProbeExecutionAuthorizationError(f"{label} is not initialized/trusted") from exc
    if type(state) is not _SkillProbeExecutionAuthorizationState:
        raise SkillProbeExecutionAuthorizationError(f"{label} is not initialized/trusted")
    return state


class SkillProbeExecutionAuthorizationCapability:
    """Verify grants and their one exact controller-registered closure."""

    __slots__ = ("__state",)

    def __new__(cls, *args: object, **kwargs: object) -> SkillProbeExecutionAuthorizationCapability:
        del args, kwargs
        raise TypeError(
            "skill-probe verification capabilities are created only by the trusted service"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("skill-probe execution authorization capability cannot be subclassed")

    def _trusted_state(self) -> _SkillProbeExecutionAuthorizationState:
        return _require_trusted_state(
            self,
            attribute="_SkillProbeExecutionAuthorizationCapability__state",
            label="authorization capability",
        )

    @property
    def repository(self) -> ArtifactRepository:
        """Return the exact repository object owned by the issuing controller."""

        return self._trusted_state().repository

    def verify_skill_probe_execution_authorization(
        self,
        ref: ArtifactRef,
    ) -> SkillProbeExecutionAuthorization:
        """Re-authenticate an issued grant without consuming its execution right."""

        state = self._trusted_state()
        checked_ref, issued = state.issued_authorization(ref)
        return state.verify_issued(checked_ref, issued)

    def verify_registered_skill_probe_execution_closure(
        self,
        authorization_ref: ArtifactRef,
        closure_ref: ArtifactRef,
    ) -> SkillProbeExecutionAuthorization:
        """Authenticate the sole closure registered for an issued grant."""

        return self._trusted_state().verify_registered_closure(
            authorization_ref,
            closure_ref,
        )


class TrustedSkillProbeExecutionAuthority:
    """Start execution and bind its result; held only by the live controller."""

    __slots__ = ("__state",)

    def __new__(cls, *args: object, **kwargs: object) -> TrustedSkillProbeExecutionAuthority:
        del args, kwargs
        raise TypeError("skill-probe execution authorities are created only by the trusted service")

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("trusted skill-probe execution authority cannot be subclassed")

    def _trusted_state(self) -> _SkillProbeExecutionAuthorizationState:
        return _require_trusted_state(
            self,
            attribute="_TrustedSkillProbeExecutionAuthority__state",
            label="skill-probe execution authority",
        )

    @property
    def repository(self) -> ArtifactRepository:
        """Return the exact repository object owned by the issuing controller."""

        return self._trusted_state().repository

    def begin_skill_probe_execution(
        self,
        ref: ArtifactRef,
    ) -> SkillProbeExecutionAuthorization:
        """Atomically and irreversibly start one issued execution right."""

        return self._trusted_state().begin(ref)

    def register_skill_probe_execution_closure(
        self,
        authorization_ref: ArtifactRef,
        closure_ref: ArtifactRef,
    ) -> SkillProbeExecutionAuthorization:
        """Atomically bind the started grant to one exact closure reference."""

        return self._trusted_state().register_closure(authorization_ref, closure_ref)


class TrustedSkillProbeExecutionAuthorizationService:
    """Issue at most one process-local grant per candidate probe head."""

    __slots__ = (
        "__by_head",
        "__capability",
        "__execution_authority",
        "__lock",
        "__state",
    )

    def __new__(
        cls,
        *args: object,
        **kwargs: object,
    ) -> TrustedSkillProbeExecutionAuthorizationService:
        del args, kwargs
        raise TypeError(
            "skill-probe authorization services are created only by controller composition"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("trusted skill-probe authorization service cannot be subclassed")

    def _trusted_state(self) -> _SkillProbeExecutionAuthorizationState:
        return _require_trusted_state(
            self,
            attribute="_TrustedSkillProbeExecutionAuthorizationService__state",
            label="skill-probe authorization service",
        )

    @property
    def verification_capability(self) -> SkillProbeExecutionAuthorizationCapability:
        self._trusted_state()
        return self.__capability

    @property
    def execution_authority(self) -> TrustedSkillProbeExecutionAuthority:
        self._trusted_state()
        return self.__execution_authority

    def _verify_current(self, authorization: SkillProbeExecutionAuthorization) -> None:
        self._trusted_state().verify_current_authorization(authorization)

    def issue(
        self,
        *,
        experiment_ref: ArtifactRef,
        protocol_ref: ArtifactRef,
        candidate_ref: ArtifactRef,
        plan_ref: ArtifactRef,
        running_probes_tail_ref: ArtifactRef,
    ) -> ArtifactRef:
        state = self._trusted_state()
        key = (_ref_key(candidate_ref), _ref_key(running_probes_tail_ref))
        with self.__lock:
            existing = self.__by_head.get(key)
            if existing is not None:
                self.__capability.verify_skill_probe_execution_authorization(existing)
                return existing
            authorization = SkillProbeExecutionAuthorization(
                execution_nonce=secrets.token_hex(32),
                experiment_ref=experiment_ref,
                protocol_ref=protocol_ref,
                candidate_ref=candidate_ref,
                plan_ref=plan_ref,
                running_probes_tail_ref=running_probes_tail_ref,
            )
            self._verify_current(authorization)
            ref = state.repository.put_json(
                authorization,
                media_type=SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE,
            )
            state.register_issued(ref, authorization)
            self.__by_head[key] = ref
            return ref


def _create_trusted_skill_probe_execution_authorization_service(
    repository: ArtifactRepository,
    *,
    verify_current: Callable[[SkillProbeExecutionAuthorization], None],
) -> TrustedSkillProbeExecutionAuthorizationService:
    """Compose the split authority for one controller; not a public API."""

    state = _SkillProbeExecutionAuthorizationState(
        repository,
        verify_current=verify_current,
    )
    capability = object.__new__(SkillProbeExecutionAuthorizationCapability)
    capability._SkillProbeExecutionAuthorizationCapability__state = state
    execution_authority = object.__new__(TrustedSkillProbeExecutionAuthority)
    execution_authority._TrustedSkillProbeExecutionAuthority__state = state
    service = object.__new__(TrustedSkillProbeExecutionAuthorizationService)
    service._TrustedSkillProbeExecutionAuthorizationService__state = state
    service._TrustedSkillProbeExecutionAuthorizationService__capability = capability
    service._TrustedSkillProbeExecutionAuthorizationService__execution_authority = (
        execution_authority
    )
    service._TrustedSkillProbeExecutionAuthorizationService__by_head = {}
    service._TrustedSkillProbeExecutionAuthorizationService__lock = Lock()
    return service


__all__ = [
    "SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE",
    "SkillProbeExecutionAuthorization",
    "SkillProbeExecutionAuthorizationError",
]
