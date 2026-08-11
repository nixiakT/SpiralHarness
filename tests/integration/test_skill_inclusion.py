from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    AttemptBudget,
    BackendResponse,
    BackendTokenUsage,
    CandidateTask,
    FrozenModelSpec,
    InferenceConfig,
    ResolvedHarness,
)
from spiral_harness.execution.materialization import HarnessMaterializer
from spiral_harness.execution.model import (
    FixedModelRunner,
    ReplayBackend,
    materialize_request,
    replay_key,
)
from spiral_harness.execution.receipts import (
    ScheduledExecutionRecord,
    execute_scheduled_attempt,
)
from spiral_harness.execution.schedule import (
    EvaluationBatchSchedule,
    EvaluationPhase,
    EvaluationSide,
    SchedulePreflightCertificate,
    preflight_attempt_budget,
    publish_schedule_preflight,
)
from spiral_harness.skills.loading import SkillDisclosureLevel
from spiral_harness.skills.package import (
    SKILL_PACKAGE_MEDIA_TYPE,
    SkillLicense,
    SkillPackage,
    SkillRule,
    SkillSourceKind,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.skill_inclusion import (
    SETTLED_SKILL_REQUEST_INCLUSION_CLAIM,
    SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
    SettledSkillRequestInclusionEvidence,
    SkillRequestInclusionError,
    publish_settled_skill_request_inclusion,
    verify_settled_skill_request_inclusion,
)

BACKEND_FINGERPRINT = "skill-inclusion-replay@sha256:fixed-v1"
BASE_PROMPT = "Solve the task and return only the final answer."


def fixed_spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="deterministic-replay",
        backend_fingerprint=BACKEND_FINGERPRINT,
        model="hosted/skill-inclusion-model",
        revision="snapshot-2026-08-12",
        tokenizer="provider/skill-inclusion-tokenizer",
        tokenizer_revision="snapshot-2026-08-12",
        runtime="skill-inclusion-worker@sha256:fixed-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=8,
            timeout_seconds=5.0,
        ),
    )


def put_skill_package(
    repository: ArtifactRepository,
    spec: FrozenModelSpec,
) -> ArtifactRef:
    package = SkillPackage(
        skill_id="verify-arithmetic",
        revision=0,
        name="Verify arithmetic",
        summary="Checks arithmetic before returning a final answer.",
        activation_guidance="Use for arithmetic questions.",
        applicability_tags=("arithmetic", "verification"),
        rules=(
            SkillRule(
                rule_id="recompute",
                instruction="Recompute the final arithmetic before answering.",
            ),
        ),
        procedure="Solve, recompute independently, then return the answer.",
        compatible_model_fingerprints=(spec.model_fingerprint,),
        runtime_fingerprints=(spec.runtime_fingerprint,),
        license=SkillLicense(
            spdx_expression="Apache-2.0",
            source_kind=SkillSourceKind.FIRST_PARTY,
            provenance_refs=(repository.put_bytes(b"skill source", media_type="text/plain"),),
            compliance_review_ref=repository.put_json(
                {"approved": True},
                media_type="application/vnd.spiral-harness.compliance-review.v1+json",
            ),
        ),
    )
    return repository.put_json(package, media_type=SKILL_PACKAGE_MEDIA_TYPE)


def put_harness(
    repository: ArtifactRepository,
    spec: FrozenModelSpec,
    *,
    prompt: str,
    skill_ref: ArtifactRef | None = None,
) -> ArtifactRef:
    prompt_ref = repository.put_bytes(prompt.encode(), media_type="text/plain")
    components = [
        HarnessComponentRef(
            name="system-prompt",
            kind=ComponentKind.PROMPT,
            artifact=prompt_ref,
        )
    ]
    if skill_ref is not None:
        components.append(
            HarnessComponentRef(
                name="verify-arithmetic",
                kind=ComponentKind.SKILL,
                artifact=skill_ref,
            )
        )
    manifest = HarnessManifest(
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        trusted_plane_version="skill-inclusion-test-v1",
        components=tuple(components),
    )
    return repository.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE)


@dataclass(frozen=True)
class InclusionRun:
    repository: ArtifactRepository
    backing_store: ArtifactStore
    spec: FrozenModelSpec
    skill_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    candidate_harness: ResolvedHarness
    schedule: EvaluationBatchSchedule
    ledger: AttemptLedger
    preflight_ref: ArtifactRef
    records: tuple[ScheduledExecutionRecord, ...]

    @property
    def receipt_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(record.receipt_ref for record in self.records)


def execute_pair(
    root: Path,
    *,
    phase: EvaluationPhase = EvaluationPhase.PROBE,
    candidate_has_skill: bool = True,
    candidate_fails: bool = False,
    candidate_poisoned: bool = False,
    paired_retry: bool = False,
    repository: ArtifactRepository | None = None,
    backing_store: ArtifactStore | None = None,
    query: int = 0,
    extra_attempts: int = 0,
) -> InclusionRun:
    if repository is None:
        backing_store = ArtifactStore(root / "cas")
        repository = backing_store
    assert backing_store is not None
    spec = fixed_spec()
    skill_ref = put_skill_package(repository, spec)
    parent_harness_ref = put_harness(
        repository,
        spec,
        prompt=BASE_PROMPT,
    )
    candidate_harness_ref = put_harness(
        repository,
        spec,
        prompt=BASE_PROMPT if candidate_has_skill else BASE_PROMPT + " Candidate revision.",
        skill_ref=skill_ref if candidate_has_skill else None,
    )
    materializer = HarnessMaterializer(repository, spec=spec)
    harnesses = {
        EvaluationSide.PARENT: materializer.materialize(parent_harness_ref),
        EvaluationSide.CANDIDATE: materializer.materialize(candidate_harness_ref),
    }
    schedule = EvaluationBatchSchedule(
        study="settled-skill-request-inclusion",
        kind="skill-request-inclusion",
        phase=phase,
        query=query,
        master_seed=20260812,
        parent_harness_id=parent_harness_ref.sha256,
        candidate_harness_id=candidate_harness_ref.sha256,
        task_ids=("arithmetic-1",),
        search_runs=(0,),
        repeat_seeds=(7,),
        max_attempts_per_cell=2 if paired_retry else 1,
        token_ceiling_per_attempt=8,
    )
    task = CandidateTask(task_id="arithmetic-1", question="What is 20 + 22?")
    responses: dict[str, BackendResponse] = {}
    for cell in schedule.iter_cells():
        response_attempts = (1,) if paired_retry else (0,)
        for attempt_index in response_attempts:
            request = materialize_request(
                task,
                harnesses[cell.side],
                seed=schedule.seed_for(cell, attempt_index=attempt_index),
            )
            if candidate_fails and cell.side is EvaluationSide.CANDIDATE:
                continue
            responses[replay_key(spec, request)] = BackendResponse(
                output='{"passed":true,"activated":true}',
                usage=BackendTokenUsage(
                    input_tokens=2,
                    output_tokens=(
                        7 if candidate_poisoned and cell.side is EvaluationSide.CANDIDATE else 1
                    ),
                ),
            )
    backend = ReplayBackend(fingerprint=BACKEND_FINGERPRINT, responses=responses)
    attempts = schedule.required_attempts + extra_attempts
    ledger = AttemptLedger(
        repository,
        ledger_id=f"skill-inclusion-ledger-{query}",
        budget=AttemptBudget(
            max_attempts=attempts,
            max_total_tokens=attempts * schedule.token_ceiling_per_attempt,
            max_tokens_per_attempt=schedule.token_ceiling_per_attempt,
        ),
    )
    preflight_ref = publish_schedule_preflight(
        repository,
        preflight_attempt_budget(schedule, ledger, spec),
    )
    runner = FixedModelRunner(spec=spec, backend=backend, attempt_ledger=ledger)
    expected_tail: ArtifactRef | None = None
    previous_receipt_ref: ArtifactRef | None = None
    records = []
    for cell in schedule.iter_cells():
        attempt_indexes = range(schedule.max_attempts_per_cell) if paired_retry else (0,)
        for attempt_index in attempt_indexes:
            record = execute_scheduled_attempt(
                runner=runner,
                schedule=schedule,
                preflight_ref=preflight_ref,
                expected_previous_ledger_tail_ref=expected_tail,
                previous_receipt_ref=previous_receipt_ref,
                cell=cell,
                attempt_index=attempt_index,
                task=task,
                harness_ref=(
                    parent_harness_ref
                    if cell.side is EvaluationSide.PARENT
                    else candidate_harness_ref
                ),
            )
            records.append(record)
            expected_tail = record.outcome_ref
            previous_receipt_ref = record.receipt_ref
    return InclusionRun(
        repository=repository,
        backing_store=backing_store,
        spec=spec,
        skill_ref=skill_ref,
        parent_harness_ref=parent_harness_ref,
        candidate_harness_ref=candidate_harness_ref,
        candidate_harness=harnesses[EvaluationSide.CANDIDATE],
        schedule=schedule,
        ledger=ledger,
        preflight_ref=preflight_ref,
        records=tuple(records),
    )


def publish(
    run: InclusionRun,
    *,
    receipt_refs: tuple[ArtifactRef, ...] | None = None,
    candidate_harness_ref: ArtifactRef | None = None,
    preflight_ref: ArtifactRef | None = None,
) -> ArtifactRef:
    return publish_settled_skill_request_inclusion(
        run.repository,
        schedule=run.schedule,
        preflight_ref=preflight_ref or run.preflight_ref,
        attempt_ledger=run.ledger,
        receipt_refs=run.receipt_refs if receipt_refs is None else receipt_refs,
        candidate_harness_ref=candidate_harness_ref or run.candidate_harness_ref,
    )


def verify(
    run: InclusionRun,
    evidence_ref: ArtifactRef,
    *,
    schedule: EvaluationBatchSchedule | None = None,
    preflight_ref: ArtifactRef | None = None,
    candidate_harness_ref: ArtifactRef | None = None,
) -> SettledSkillRequestInclusionEvidence:
    return verify_settled_skill_request_inclusion(
        run.repository,
        evidence_ref=evidence_ref,
        schedule=run.schedule if schedule is None else schedule,
        preflight_ref=run.preflight_ref if preflight_ref is None else preflight_ref,
        attempt_ledger=run.ledger,
        candidate_harness_ref=(
            run.candidate_harness_ref if candidate_harness_ref is None else candidate_harness_ref
        ),
    )


def nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(nested_keys(item) for item in value))
    return set()


def test_publish_derives_exact_inclusion_from_settled_candidate_request(tmp_path: Path) -> None:
    run = execute_pair(tmp_path)

    evidence_ref = publish(run)
    evidence = run.repository.get_json(
        evidence_ref,
        SettledSkillRequestInclusionEvidence,
    )

    assert evidence_ref.media_type == SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE
    assert evidence.claim == SETTLED_SKILL_REQUEST_INCLUSION_CLAIM
    assert evidence.schedule == run.schedule
    assert evidence.preflight_ref == run.preflight_ref
    assert evidence.model_spec_fingerprint == run.spec.fingerprint
    assert evidence.candidate_harness_ref == run.candidate_harness_ref
    assert evidence.usage.cell_count == run.schedule.cell_count == 2
    expected_receipt_refs = tuple(
        record.receipt_ref
        for record in sorted(
            run.records,
            key=lambda record: (record.cell.fingerprint, record.attempt_index),
        )
    )
    assert evidence.usage.receipt_refs == expected_receipt_refs
    assert len(evidence.observations) == 1
    observation = evidence.observations[0]
    assert observation.cell.side is EvaluationSide.CANDIDATE
    assert observation.attempt_index == 0
    assert observation.receipt_ref == next(
        record.receipt_ref for record in run.records if record.cell.side is EvaluationSide.CANDIDATE
    )
    assert evidence.skill_disclosure == run.candidate_harness.skill_disclosure
    assert evidence.skill_disclosure.package_ref == run.skill_ref
    assert evidence.skill_disclosure.level is SkillDisclosureLevel.RULES
    payload = run.repository.get_json(evidence_ref)
    assert not nested_keys(payload).intersection({"activated", "adhered", "passed"})


def test_paired_retry_observes_only_the_candidate_terminal_settled_attempt(
    tmp_path: Path,
) -> None:
    run = execute_pair(tmp_path, paired_retry=True)

    evidence_ref = publish(run)
    evidence = run.repository.get_json(
        evidence_ref,
        SettledSkillRequestInclusionEvidence,
    )

    assert evidence.usage.attempt_count == 4
    assert evidence.usage.burned_attempts == evidence.usage.settled_attempts == 2
    assert len(evidence.observations) == 1
    observation = evidence.observations[0]
    terminal = next(
        record
        for record in run.records
        if record.cell.side is EvaluationSide.CANDIDATE and record.attempt_index == 1
    )
    assert observation.attempt_index == 1
    assert observation.receipt_ref == terminal.receipt_ref
    assert observation.outcome_ref == terminal.outcome_ref
    assert observation.execution_ref == terminal.execution_ref


def test_verify_rederives_the_exact_published_artifact(tmp_path: Path) -> None:
    run = execute_pair(tmp_path)
    evidence_ref = publish(run)
    published = run.repository.get_json(
        evidence_ref,
        SettledSkillRequestInclusionEvidence,
    )

    verified = verify(run, evidence_ref)

    assert verified == published


def test_verify_is_read_only_when_evidence_put_hook_is_armed(tmp_path: Path) -> None:
    backing_store = ArtifactStore(tmp_path / "cas")
    repository = AdvancingRepository(backing_store)
    run = execute_pair(
        tmp_path,
        repository=repository,
        backing_store=backing_store,
        extra_attempts=1,
    )
    evidence_ref = publish(run)
    repository.ledger = run.ledger
    repository.armed = True
    ledger_before = run.ledger.state()

    verified = verify(run, evidence_ref)

    assert verified == repository.get_json(
        evidence_ref,
        SettledSkillRequestInclusionEvidence,
    )
    assert repository.armed is True
    assert run.ledger.state() == ledger_before


def test_verify_rejects_tampered_published_evidence_bytes(tmp_path: Path) -> None:
    run = execute_pair(tmp_path)
    evidence_ref = publish(run)
    run.backing_store.path_for(evidence_ref).write_bytes(b"x" * evidence_ref.size)

    with pytest.raises(SkillRequestInclusionError, match="cannot be verified"):
        verify(run, evidence_ref)


def test_verify_rejects_a_historical_ledger_after_an_unreceipted_burn(
    tmp_path: Path,
) -> None:
    run = execute_pair(tmp_path, extra_attempts=1)
    evidence_ref = publish(run)
    historical_tail = run.ledger.tail_ref
    assert historical_tail is not None
    reservation_ref = run.ledger.reserve(
        task_fingerprint="a" * 64,
        execution_fingerprint="b" * 64,
        request_sha256="c" * 64,
        token_ceiling=1,
    )
    run.ledger.burn(reservation_ref, error_class="unreceipted-trailing-burn")
    historical = AttemptLedger(
        run.repository,
        ledger_id=run.ledger.ledger_id,
        budget=run.ledger.budget,
        tail_ref=historical_tail,
    )

    with pytest.raises(SkillRequestInclusionError, match="writer epoch"):
        verify_settled_skill_request_inclusion(
            run.repository,
            evidence_ref=evidence_ref,
            schedule=run.schedule,
            preflight_ref=run.preflight_ref,
            attempt_ledger=historical,
            candidate_harness_ref=run.candidate_harness_ref,
        )


@pytest.mark.parametrize(
    "target",
    ["model-spec-fingerprint", "observation-attempt", "request-sha"],
)
def test_schema_valid_forged_artifacts_fail_trusted_rederivation(
    tmp_path: Path,
    target: str,
) -> None:
    run = execute_pair(tmp_path / target)
    evidence_ref = publish(run)
    evidence = run.repository.get_json(
        evidence_ref,
        SettledSkillRequestInclusionEvidence,
    )
    values = evidence.model_dump(mode="python", round_trip=True, warnings="none")
    if target == "model-spec-fingerprint":
        values["model_spec_fingerprint"] = "f" * 64
    else:
        observation = evidence.observations[0]
        observation_values = observation.model_dump(
            mode="python",
            round_trip=True,
            warnings="none",
        )
        if target == "observation-attempt":
            observation_values["attempt_index"] = observation.attempt_index + 1
        else:
            observation_values["request_sha256"] = "e" * 64
        values["observations"] = (observation_values,)
    forged = SettledSkillRequestInclusionEvidence.model_validate(values, strict=True)
    assert forged != evidence
    forged_ref = run.repository.put_json(
        forged,
        media_type=SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
    )
    assert forged_ref != evidence_ref

    with pytest.raises(SkillRequestInclusionError, match="trusted re-derivation"):
        verify(run, forged_ref)


@pytest.mark.parametrize("target", ["schedule", "preflight", "candidate-harness"])
def test_verify_rejects_wrong_expected_context(tmp_path: Path, target: str) -> None:
    run = execute_pair(tmp_path / target)
    evidence_ref = publish(run)
    updates: dict[str, object] = {}
    if target == "schedule":
        updates["schedule"] = run.schedule.model_copy(update={"query": 9})
    elif target == "preflight":
        updates["preflight_ref"] = ArtifactRef(
            sha256="d" * 64,
            size=1,
            media_type=run.preflight_ref.media_type,
        )
    else:
        updates["candidate_harness_ref"] = run.parent_harness_ref

    with pytest.raises(SkillRequestInclusionError, match="expected"):
        verify(run, evidence_ref, **updates)


def test_evidence_schema_fixes_claim_and_reference_media(tmp_path: Path) -> None:
    run = execute_pair(tmp_path)
    evidence_ref = publish(run)
    evidence = run.repository.get_json(
        evidence_ref,
        SettledSkillRequestInclusionEvidence,
    )
    values = evidence.model_dump(mode="python", round_trip=True, warnings="none")

    with pytest.raises(ValidationError, match="claim"):
        SettledSkillRequestInclusionEvidence.model_validate({**values, "claim": "skill-activated"})
    wrong_preflight = evidence.preflight_ref.model_copy(update={"media_type": "application/json"})
    with pytest.raises(ValidationError, match="preflight_ref"):
        SettledSkillRequestInclusionEvidence.model_validate(
            {**values, "preflight_ref": wrong_preflight}
        )


@pytest.mark.parametrize("phase", [EvaluationPhase.GATE, EvaluationPhase.SEALED])
def test_non_probe_schedules_are_rejected(tmp_path: Path, phase: EvaluationPhase) -> None:
    run = execute_pair(tmp_path / phase.value, phase=phase)

    with pytest.raises(SkillRequestInclusionError, match="PROBE"):
        publish(run)


def test_prompt_only_candidate_is_not_skill_inclusion(tmp_path: Path) -> None:
    run = execute_pair(tmp_path, candidate_has_skill=False)

    with pytest.raises(SkillRequestInclusionError, match="does not contain a skill"):
        publish(run)


@pytest.mark.parametrize("case", ["missing", "duplicate", "foreign"])
def test_incomplete_or_foreign_receipt_sets_fail_closed(tmp_path: Path, case: str) -> None:
    backing_store = ArtifactStore(tmp_path / "cas")
    run = execute_pair(
        tmp_path,
        repository=backing_store,
        backing_store=backing_store,
    )
    if case == "missing":
        refs = run.receipt_refs[:-1]
    elif case == "duplicate":
        refs = (*run.receipt_refs, run.receipt_refs[0])
    else:
        foreign = execute_pair(
            tmp_path,
            repository=backing_store,
            backing_store=backing_store,
            query=1,
        )
        refs = (*run.receipt_refs, foreign.receipt_refs[0])

    with pytest.raises(SkillRequestInclusionError, match="receipt replay failed"):
        publish(run, receipt_refs=refs)


@pytest.mark.parametrize("case", ["wrong-harness", "wrong-media"])
def test_candidate_harness_must_be_the_exact_scheduled_manifest(
    tmp_path: Path,
    case: str,
) -> None:
    run = execute_pair(tmp_path / case)
    candidate_ref = (
        run.parent_harness_ref
        if case == "wrong-harness"
        else run.candidate_harness_ref.model_copy(update={"media_type": "application/json"})
    )

    with pytest.raises(SkillRequestInclusionError, match="candidate harness"):
        publish(run, candidate_harness_ref=candidate_ref)


@pytest.mark.parametrize("target", ["package", "execution", "preflight"])
def test_cas_tampering_fails_closed(tmp_path: Path, target: str) -> None:
    run = execute_pair(tmp_path / target)
    candidate_record = next(
        record for record in run.records if record.cell.side is EvaluationSide.CANDIDATE
    )
    target_ref = {
        "package": run.skill_ref,
        "execution": candidate_record.execution_ref,
        "preflight": run.preflight_ref,
    }[target]
    run.backing_store.path_for(target_ref).write_bytes(b"x" * target_ref.size)

    with pytest.raises(SkillRequestInclusionError):
        publish(run)


def test_foreign_preflight_model_spec_is_rejected(tmp_path: Path) -> None:
    run = execute_pair(tmp_path)
    preflight = run.repository.get_json(run.preflight_ref, SchedulePreflightCertificate)
    foreign_spec = run.spec.model_copy(update={"revision": "snapshot-foreign"})
    foreign_preflight_ref = run.repository.put_json(
        preflight.model_copy(update={"model_spec": foreign_spec}),
        media_type=run.preflight_ref.media_type,
    )

    with pytest.raises(SkillRequestInclusionError, match="receipt replay failed"):
        publish(run, preflight_ref=foreign_preflight_ref)


def test_independent_verifier_rejects_forged_preflight_shape(tmp_path: Path) -> None:
    run = execute_pair(tmp_path)
    preflight = run.repository.get_json(run.preflight_ref, SchedulePreflightCertificate)
    values = preflight.model_dump(mode="python", round_trip=True, warnings="none")
    ceiling = preflight.token_ceiling_per_attempt + 1
    required_tokens = preflight.required_attempts * ceiling
    values.update(
        token_ceiling_per_attempt=ceiling,
        required_tokens=required_tokens,
        available_tokens=max(preflight.available_tokens, required_tokens),
    )
    forged = SchedulePreflightCertificate.model_validate(values, strict=True)
    forged_ref = publish_schedule_preflight(run.repository, forged)

    with pytest.raises(SkillRequestInclusionError, match="preflight shape differs"):
        publish(run, preflight_ref=forged_ref)


def test_candidate_terminal_failure_is_not_settled_inclusion(tmp_path: Path) -> None:
    run = execute_pair(tmp_path, candidate_fails=True)

    with pytest.raises(SkillRequestInclusionError, match="must end in a settled attempt"):
        publish(run)


def test_a_poisoned_attempt_invalidates_the_complete_inclusion_batch(tmp_path: Path) -> None:
    run = execute_pair(tmp_path, candidate_poisoned=True)
    assert run.ledger.state().poisoned is True

    with pytest.raises(SkillRequestInclusionError, match="poisoned attempt"):
        publish(run)


class AdvancingRepository:
    """Test repository that advances the ledger during evidence publication."""

    def __init__(self, backing_store: ArtifactStore) -> None:
        self.backing_store = backing_store
        self.ledger: AttemptLedger | None = None
        self.armed = False
        self.last_evidence_ref: ArtifactRef | None = None

    def put_bytes(
        self,
        data: bytes | bytearray | memoryview,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        return self.backing_store.put_bytes(data, media_type=media_type)

    def put_json(self, value: Any, *, media_type: str = "application/json") -> ArtifactRef:
        ref = self.backing_store.put_json(value, media_type=media_type)
        if media_type == SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE:
            self.last_evidence_ref = ref
        if self.armed and media_type == SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE:
            self.armed = False
            assert self.ledger is not None
            self.ledger.reserve(
                task_fingerprint="a" * 64,
                execution_fingerprint="b" * 64,
                request_sha256="c" * 64,
                token_ceiling=1,
            )
        return ref

    def get_bytes(self, ref_or_digest: ArtifactRef | str) -> bytes:
        return self.backing_store.get_bytes(ref_or_digest)

    def get_json(
        self,
        ref_or_digest: ArtifactRef | str,
        model_type: type[Any] | None = None,
    ) -> Any:
        return self.backing_store.get_json(ref_or_digest, model_type)


def test_ledger_advance_during_publication_is_rejected(tmp_path: Path) -> None:
    backing_store = ArtifactStore(tmp_path / "cas")
    repository = AdvancingRepository(backing_store)
    run = execute_pair(
        tmp_path,
        repository=repository,
        backing_store=backing_store,
        extra_attempts=1,
    )
    repository.ledger = run.ledger
    repository.armed = True
    historical_tail = run.ledger.tail_ref
    assert historical_tail is not None

    with pytest.raises(SkillRequestInclusionError, match="changed"):
        publish(run)

    assert repository.last_evidence_ref is not None
    historical = AttemptLedger(
        repository,
        ledger_id=run.ledger.ledger_id,
        budget=run.ledger.budget,
        tail_ref=historical_tail,
    )
    with pytest.raises(SkillRequestInclusionError, match="writer epoch"):
        verify_settled_skill_request_inclusion(
            repository,
            evidence_ref=repository.last_evidence_ref,
            schedule=run.schedule,
            preflight_ref=run.preflight_ref,
            attempt_ledger=historical,
            candidate_harness_ref=run.candidate_harness_ref,
        )
