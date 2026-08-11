"""Receipt-backed proof that a settled request contained one exact skill.

This module deliberately proves only request inclusion.  It does not infer
that the model activated, followed, or benefited from the disclosed rules.
The evidence is derived from the persisted request and accounting chain; no
candidate-authored boolean or model output participates in the claim.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ImmutableModel,
    Sha256,
)
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptDisposition,
    AttemptOutcome,
    ExecutionStatus,
    ModelExecution,
)
from spiral_harness.execution.materialization import (
    HarnessMaterializationError,
    HarnessMaterializer,
)
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceipt,
    TrustedExecutionUsage,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    EvaluationCellKey,
    EvaluationPhase,
    EvaluationSide,
    SchedulePreflightCertificate,
)
from spiral_harness.skills.loading import SkillDisclosure, SkillDisclosureLevel
from spiral_harness.storage.protocol import ArtifactRepository

SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.settled-skill-request-inclusion.v1+json"
)
SETTLED_SKILL_REQUEST_INCLUSION_CLAIM = "skill-disclosure-present-in-settled-request"


class SkillRequestInclusionError(ValueError):
    """Raised when settled request inclusion cannot be proven exactly."""


class SettledSkillRequestObservation(ImmutableModel):
    """The terminal settled attempt for one candidate-side logical cell."""

    schema_version: Literal["1"] = "1"
    cell: EvaluationCellKey
    attempt_index: Annotated[int, Field(ge=0, strict=True)]
    receipt_ref: ArtifactRef
    outcome_ref: ArtifactRef
    execution_ref: ArtifactRef
    request_sha256: Sha256

    @model_validator(mode="after")
    def references_and_side_are_exact(self) -> Self:
        if self.cell.side is not EvaluationSide.CANDIDATE:
            raise ValueError("a skill request observation must belong to the candidate side")
        expected_media_types = (
            ("receipt_ref", EXECUTION_RECEIPT_MEDIA_TYPE),
            ("outcome_ref", ATTEMPT_OUTCOME_MEDIA_TYPE),
            ("execution_ref", MODEL_EXECUTION_MEDIA_TYPE),
        )
        for field_name, expected in expected_media_types:
            if getattr(self, field_name).media_type != expected:
                raise ValueError(f"{field_name} declares the wrong media type")
        return self


class SettledSkillRequestInclusionEvidence(ImmutableModel):
    """Exact request-inclusion fact derived from a complete paired replay."""

    schema_version: Literal["1"] = "1"
    claim: Literal["skill-disclosure-present-in-settled-request"] = (
        SETTLED_SKILL_REQUEST_INCLUSION_CLAIM
    )
    schedule: EvaluationBatchSchedule
    preflight_ref: ArtifactRef
    model_spec_fingerprint: Sha256
    candidate_harness_ref: ArtifactRef
    usage: TrustedExecutionUsage
    skill_disclosure: SkillDisclosure
    observations: Annotated[
        tuple[SettledSkillRequestObservation, ...],
        Field(min_length=1),
    ]

    @field_validator("observations")
    @classmethod
    def canonicalize_observations(
        cls,
        observations: tuple[SettledSkillRequestObservation, ...],
    ) -> tuple[SettledSkillRequestObservation, ...]:
        return tuple(sorted(observations, key=lambda item: item.cell.fingerprint))

    @model_validator(mode="after")
    def evidence_shape_is_exact(self) -> Self:
        if self.schedule.phase is not EvaluationPhase.PROBE:
            raise ValueError("settled skill request inclusion requires a PROBE schedule")
        if self.preflight_ref.media_type != SCHEDULE_PREFLIGHT_MEDIA_TYPE:
            raise ValueError("preflight_ref declares the wrong media type")
        if self.candidate_harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError("candidate_harness_ref declares the wrong media type")
        if self.schedule.candidate_harness_id != self.candidate_harness_ref.sha256:
            raise ValueError("candidate harness differs from the frozen schedule")
        if self.usage.schedule_fingerprint != self.schedule.fingerprint:
            raise ValueError("trusted usage belongs to another schedule")
        if self.usage.cell_count != self.schedule.cell_count:
            raise ValueError("trusted usage does not cover the complete schedule")
        if self.usage.poisoned_attempts:
            raise ValueError("poisoned attempts invalidate request-inclusion evidence")
        if self.skill_disclosure.level is not SkillDisclosureLevel.RULES:
            raise ValueError("settled skill request inclusion requires RULES disclosure")

        expected_cells = {
            cell.fingerprint: cell
            for cell in self.schedule.iter_cells()
            if cell.side is EvaluationSide.CANDIDATE
        }
        observed_cells = {
            observation.cell.fingerprint: observation for observation in self.observations
        }
        if len(observed_cells) != len(self.observations):
            raise ValueError("observations must not duplicate candidate logical cells")
        if set(observed_cells) != set(expected_cells):
            raise ValueError("observations do not cover the exact candidate schedule")
        for fingerprint, observation in observed_cells.items():
            if observation.cell != expected_cells[fingerprint]:
                raise ValueError("an observation relabels its candidate logical cell")

        receipt_digests = tuple(item.receipt_ref.sha256 for item in self.observations)
        outcome_digests = tuple(item.outcome_ref.sha256 for item in self.observations)
        execution_digests = tuple(item.execution_ref.sha256 for item in self.observations)
        if len(receipt_digests) != len(set(receipt_digests)):
            raise ValueError("observations must not reuse a receipt")
        if len(outcome_digests) != len(set(outcome_digests)):
            raise ValueError("observations must not reuse an outcome")
        if len(execution_digests) != len(set(execution_digests)):
            raise ValueError("observations must not reuse an execution")
        usage_receipts = {ref.sha256 for ref in self.usage.receipt_refs}
        if not set(receipt_digests).issubset(usage_receipts):
            raise ValueError("an observation cites a receipt outside trusted usage")
        return self


def _require_repository(repository: ArtifactRepository) -> ArtifactRepository:
    if not isinstance(repository, ArtifactRepository):
        raise SkillRequestInclusionError("repository must implement ArtifactRepository")
    return repository


def _validate_ref(
    ref: ArtifactRef,
    *,
    expected_media_type: str,
    label: str,
) -> ArtifactRef:
    try:
        checked = ArtifactRef.model_validate(ref, strict=True)
    except Exception as exc:
        raise SkillRequestInclusionError(f"{label} reference is malformed") from exc
    if checked.media_type != expected_media_type:
        raise SkillRequestInclusionError(f"{label} reference declares the wrong media type")
    return checked


def _load_exact[ModelT: BaseModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    *,
    expected_media_type: str,
    label: str,
) -> ModelT:
    checked_ref = _validate_ref(
        ref,
        expected_media_type=expected_media_type,
        label=label,
    )
    try:
        payload = repository.get_bytes(checked_ref)
        loaded = repository.get_json(checked_ref, model_type)
        checked = model_type.model_validate(loaded, strict=True)
        canonical = canonical_json_bytes(checked)
    except Exception as exc:
        raise SkillRequestInclusionError(f"{label} artifact cannot be verified") from exc
    if len(payload) != checked_ref.size:
        raise SkillRequestInclusionError(f"{label} artifact size does not match its reference")
    if sha256_bytes(canonical) != checked_ref.sha256:
        raise SkillRequestInclusionError(
            f"{label} canonical content does not match its reference digest"
        )
    if payload != canonical:
        raise SkillRequestInclusionError(
            f"{label} artifact is not in its canonical typed representation"
        )
    return checked


def _receipt_refs(receipt_refs: Iterable[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    if isinstance(receipt_refs, str | bytes | bytearray):
        raise SkillRequestInclusionError("receipt_refs must be an iterable of ArtifactRef values")
    try:
        refs = tuple(receipt_refs)
    except TypeError as exc:
        raise SkillRequestInclusionError("receipt_refs must be iterable") from exc
    if not refs:
        raise SkillRequestInclusionError("receipt_refs must not be empty")
    return refs


def _load_preflight(
    repository: ArtifactRepository,
    preflight_ref: ArtifactRef,
    schedule: EvaluationBatchSchedule,
) -> tuple[ArtifactRef, SchedulePreflightCertificate]:
    checked_ref = _validate_ref(
        preflight_ref,
        expected_media_type=SCHEDULE_PREFLIGHT_MEDIA_TYPE,
        label="preflight",
    )
    preflight = _load_exact(
        repository,
        checked_ref,
        SchedulePreflightCertificate,
        expected_media_type=SCHEDULE_PREFLIGHT_MEDIA_TYPE,
        label="preflight",
    )
    if preflight.fingerprint != checked_ref.sha256:
        raise SkillRequestInclusionError(
            "preflight reference does not match its canonical certificate fingerprint"
        )
    if not preflight.binds_schedule(schedule):
        raise SkillRequestInclusionError("preflight shape differs from the frozen schedule")
    return checked_ref, preflight


def _terminal_candidate_receipts(
    repository: ArtifactRepository,
    usage: TrustedExecutionUsage,
) -> tuple[tuple[ArtifactRef, ExecutionReceipt], ...]:
    grouped: dict[str, list[tuple[ArtifactRef, ExecutionReceipt]]] = defaultdict(list)
    for receipt_ref in usage.receipt_refs:
        receipt = _load_exact(
            repository,
            receipt_ref,
            ExecutionReceipt,
            expected_media_type=EXECUTION_RECEIPT_MEDIA_TYPE,
            label="execution receipt",
        )
        if receipt.cell.side is EvaluationSide.CANDIDATE:
            grouped[receipt.cell_fingerprint].append((receipt_ref, receipt))
    terminals: list[tuple[ArtifactRef, ExecutionReceipt]] = []
    for attempts in grouped.values():
        attempts.sort(key=lambda item: item[1].attempt_index)
        terminals.append(attempts[-1])
    return tuple(sorted(terminals, key=lambda item: item[1].cell_fingerprint))


def _derive_observations(
    repository: ArtifactRepository,
    *,
    candidate_harness_ref: ArtifactRef,
    preflight: SchedulePreflightCertificate,
    usage: TrustedExecutionUsage,
) -> tuple[SkillDisclosure, tuple[SettledSkillRequestObservation, ...]]:
    materializer = HarnessMaterializer(repository, spec=preflight.model_spec)
    common_disclosure: SkillDisclosure | None = None
    observations: list[SettledSkillRequestObservation] = []
    for receipt_ref, receipt in _terminal_candidate_receipts(repository, usage):
        outcome = _load_exact(
            repository,
            receipt.outcome_ref,
            AttemptOutcome,
            expected_media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
            label="candidate terminal outcome",
        )
        execution = _load_exact(
            repository,
            receipt.execution_ref,
            ModelExecution,
            expected_media_type=MODEL_EXECUTION_MEDIA_TYPE,
            label="candidate terminal execution",
        )
        if outcome.disposition is not AttemptDisposition.SETTLED:
            raise SkillRequestInclusionError(
                "every candidate logical cell must end in a settled attempt"
            )
        if outcome.execution_ref != receipt.execution_ref:
            raise SkillRequestInclusionError(
                "candidate terminal outcome does not bind the receipt execution"
            )
        if execution.status is not ExecutionStatus.COMPLETED:
            raise SkillRequestInclusionError(
                "a settled candidate request must have a completed execution"
            )
        try:
            resolved = materializer.verify_execution_request(candidate_harness_ref, execution)
        except HarnessMaterializationError as exc:
            raise SkillRequestInclusionError(
                f"candidate request cannot be replayed from the exact harness: {exc}"
            ) from exc
        disclosure = resolved.skill_disclosure
        if disclosure is None:
            raise SkillRequestInclusionError(
                "candidate settled request does not contain a skill disclosure"
            )
        if disclosure.level is not SkillDisclosureLevel.RULES:
            raise SkillRequestInclusionError(
                "candidate settled request does not contain the fixed RULES disclosure"
            )
        if common_disclosure is None:
            common_disclosure = disclosure
        elif disclosure != common_disclosure:
            raise SkillRequestInclusionError(
                "candidate settled requests do not contain one common skill disclosure"
            )
        observations.append(
            SettledSkillRequestObservation(
                cell=receipt.cell,
                attempt_index=receipt.attempt_index,
                receipt_ref=receipt_ref,
                outcome_ref=receipt.outcome_ref,
                execution_ref=receipt.execution_ref,
                request_sha256=execution.request_sha256,
            )
        )
    if common_disclosure is None:
        raise SkillRequestInclusionError("the candidate schedule has no settled skill request")
    return common_disclosure, tuple(observations)


def _verify_closing_ledger(
    attempt_ledger: AttemptLedger,
    *,
    preflight: SchedulePreflightCertificate,
    usage: TrustedExecutionUsage,
) -> None:
    try:
        state = attempt_ledger.state()
    except Exception as exc:
        raise SkillRequestInclusionError("live attempt ledger cannot be reverified") from exc
    expected_tail = usage.ledger_tail_refs[0]
    changed = (
        state.tail_ref != expected_tail
        or state.ledger_id != preflight.ledger_id
        or state.writer_epoch_id != preflight.writer_epoch_id
        or state.budget.fingerprint != preflight.budget_fingerprint
        or state.pending_reservation_ref is not None
        or state.poisoned
    )
    if changed:
        raise SkillRequestInclusionError(
            "live attempt ledger changed while request-inclusion evidence was derived"
        )


def _derive_evidence(
    repository: ArtifactRepository,
    *,
    schedule: EvaluationBatchSchedule,
    preflight_ref: ArtifactRef,
    attempt_ledger: AttemptLedger,
    receipt_refs: Iterable[ArtifactRef],
    candidate_harness_ref: ArtifactRef,
) -> tuple[SettledSkillRequestInclusionEvidence, SchedulePreflightCertificate]:
    """Replay trusted state and derive evidence without writing to the repository."""

    checked_repository = _require_repository(repository)
    try:
        checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    except Exception as exc:
        raise SkillRequestInclusionError("schedule is malformed") from exc
    if checked_schedule.phase is not EvaluationPhase.PROBE:
        raise SkillRequestInclusionError("settled skill request inclusion requires PROBE phase")
    checked_harness_ref = _validate_ref(
        candidate_harness_ref,
        expected_media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        label="candidate harness",
    )
    if checked_schedule.candidate_harness_id != checked_harness_ref.sha256:
        raise SkillRequestInclusionError("candidate harness differs from the frozen schedule")
    if type(attempt_ledger) is not AttemptLedger:
        raise SkillRequestInclusionError("attempt_ledger must be an exact AttemptLedger")
    if attempt_ledger.repository is not checked_repository:
        raise SkillRequestInclusionError("attempt ledger uses a different artifact repository")
    checked_receipt_refs = _receipt_refs(receipt_refs)
    checked_preflight_ref, preflight = _load_preflight(
        checked_repository,
        preflight_ref,
        checked_schedule,
    )
    try:
        usage = replay_trusted_usage(
            checked_repository,
            schedule=checked_schedule,
            preflight_ref=checked_preflight_ref,
            attempt_ledger=attempt_ledger,
            receipt_refs=checked_receipt_refs,
        )
    except Exception as exc:
        raise SkillRequestInclusionError(f"complete paired receipt replay failed: {exc}") from exc

    disclosure, observations = _derive_observations(
        checked_repository,
        candidate_harness_ref=checked_harness_ref,
        preflight=preflight,
        usage=usage,
    )
    try:
        evidence = SettledSkillRequestInclusionEvidence(
            schedule=checked_schedule,
            preflight_ref=checked_preflight_ref,
            model_spec_fingerprint=preflight.model_spec.fingerprint,
            candidate_harness_ref=checked_harness_ref,
            usage=usage,
            skill_disclosure=disclosure,
            observations=observations,
        )
    except Exception as exc:
        raise SkillRequestInclusionError("derived request-inclusion evidence is invalid") from exc
    return evidence, preflight


def publish_settled_skill_request_inclusion(
    repository: ArtifactRepository,
    *,
    schedule: EvaluationBatchSchedule,
    preflight_ref: ArtifactRef,
    attempt_ledger: AttemptLedger,
    receipt_refs: Iterable[ArtifactRef],
    candidate_harness_ref: ArtifactRef,
) -> ArtifactRef:
    """Publish a conservative inclusion artifact from settled candidate requests.

    Complete paired receipt and live-ledger replay happens before any
    observation is derived.  The ledger is checked again after the immutable
    artifact is written and read back, closing the in-process publication
    window for every writer sharing the exact ``AttemptLedger`` instance.
    """

    checked_repository = _require_repository(repository)
    evidence, preflight = _derive_evidence(
        checked_repository,
        schedule=schedule,
        preflight_ref=preflight_ref,
        attempt_ledger=attempt_ledger,
        receipt_refs=receipt_refs,
        candidate_harness_ref=candidate_harness_ref,
    )
    try:
        published_ref = checked_repository.put_json(
            evidence,
            media_type=SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
        )
    except Exception as exc:
        raise SkillRequestInclusionError(
            "settled skill request inclusion could not be published"
        ) from exc
    checked_published_ref = _validate_ref(
        published_ref,
        expected_media_type=SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
        label="published settled skill request inclusion",
    )
    loaded = _load_exact(
        checked_repository,
        checked_published_ref,
        SettledSkillRequestInclusionEvidence,
        expected_media_type=SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
        label="published settled skill request inclusion",
    )
    if loaded != evidence:
        raise SkillRequestInclusionError(
            "published settled skill request inclusion changed content"
        )
    _verify_closing_ledger(
        attempt_ledger,
        preflight=preflight,
        usage=evidence.usage,
    )
    return checked_published_ref


def verify_settled_skill_request_inclusion(
    repository: ArtifactRepository,
    *,
    evidence_ref: ArtifactRef,
    schedule: EvaluationBatchSchedule,
    preflight_ref: ArtifactRef,
    attempt_ledger: AttemptLedger,
    candidate_harness_ref: ArtifactRef,
) -> SettledSkillRequestInclusionEvidence:
    """Re-derive one inclusion artifact under an exact expected context.

    A matching media type and schema are not producer authority.  Verification
    first loads the claimed artifact exactly, then requires its frozen context
    to equal caller-held expectations.  Finally it replays the artifact's
    complete receipt closure through a read-only trusted derivation and accepts
    only byte-identical, content-address-identical results.
    """

    checked_repository = _require_repository(repository)
    checked_evidence_ref = _validate_ref(
        evidence_ref,
        expected_media_type=SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
        label="settled skill request inclusion evidence",
    )
    evidence = _load_exact(
        checked_repository,
        checked_evidence_ref,
        SettledSkillRequestInclusionEvidence,
        expected_media_type=SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
        label="settled skill request inclusion evidence",
    )
    try:
        checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    except Exception as exc:
        raise SkillRequestInclusionError("expected schedule is malformed") from exc
    checked_preflight_ref = _validate_ref(
        preflight_ref,
        expected_media_type=SCHEDULE_PREFLIGHT_MEDIA_TYPE,
        label="expected preflight",
    )
    checked_harness_ref = _validate_ref(
        candidate_harness_ref,
        expected_media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        label="expected candidate harness",
    )
    if evidence.schedule != checked_schedule:
        raise SkillRequestInclusionError("inclusion evidence does not match the expected schedule")
    if evidence.preflight_ref != checked_preflight_ref:
        raise SkillRequestInclusionError("inclusion evidence does not match the expected preflight")
    if evidence.candidate_harness_ref != checked_harness_ref:
        raise SkillRequestInclusionError(
            "inclusion evidence does not match the expected candidate harness"
        )

    derived, preflight = _derive_evidence(
        checked_repository,
        schedule=checked_schedule,
        preflight_ref=checked_preflight_ref,
        attempt_ledger=attempt_ledger,
        receipt_refs=evidence.usage.receipt_refs,
        candidate_harness_ref=checked_harness_ref,
    )
    derived_payload = canonical_json_bytes(derived)
    derived_ref = ArtifactRef(
        sha256=sha256_bytes(derived_payload),
        size=len(derived_payload),
        media_type=SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
    )
    if derived_ref != checked_evidence_ref:
        raise SkillRequestInclusionError(
            "inclusion evidence reference does not match trusted re-derivation"
        )
    if derived_payload != canonical_json_bytes(evidence):
        raise SkillRequestInclusionError(
            "inclusion evidence content does not match trusted re-derivation"
        )
    _verify_closing_ledger(
        attempt_ledger,
        preflight=preflight,
        usage=derived.usage,
    )
    return derived


__all__ = [
    "SETTLED_SKILL_REQUEST_INCLUSION_CLAIM",
    "SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE",
    "SettledSkillRequestInclusionEvidence",
    "SettledSkillRequestObservation",
    "SkillRequestInclusionError",
    "publish_settled_skill_request_inclusion",
    "verify_settled_skill_request_inclusion",
]
