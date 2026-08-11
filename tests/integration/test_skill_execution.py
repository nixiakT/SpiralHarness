from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

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
    ModelExecution,
    ResolvedHarness,
)
from spiral_harness.execution.materialization import (
    HarnessMaterializationError,
    HarnessMaterializer,
)
from spiral_harness.execution.model import (
    FixedModelRunner,
    ReplayBackend,
    materialize_request,
    paired_execution_fingerprint,
    replay_key,
)
from spiral_harness.execution.receipts import (
    ExecutionReceiptIntegrityError,
    ScheduledExecutionRecord,
    execute_scheduled_attempt,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    EvaluationBatchSchedule,
    EvaluationPhase,
    EvaluationSide,
    preflight_attempt_budget,
    publish_schedule_preflight,
)
from spiral_harness.skills.loading import SkillDisclosureLevel, SkillPackageLoader
from spiral_harness.skills.package import (
    SKILL_PACKAGE_MEDIA_TYPE,
    SkillExample,
    SkillLicense,
    SkillPackage,
    SkillRule,
    SkillSourceKind,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.skill_inclusion import (
    SETTLED_SKILL_REQUEST_INCLUSION_CLAIM,
    SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
    SettledSkillRequestInclusionEvidence,
    publish_settled_skill_request_inclusion,
    verify_settled_skill_request_inclusion,
)

BACKEND_FINGERPRINT = "skill-integration-replay@sha256:fixed-v1"
BASE_PROMPT = "Solve the problem and return only the final answer."


def fixed_spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="deterministic-replay",
        backend_fingerprint=BACKEND_FINGERPRINT,
        model="hosted/skill-integration-model",
        revision="snapshot-2026-08-12",
        tokenizer="provider/skill-integration-tokenizer",
        tokenizer_revision="snapshot-2026-08-12",
        runtime="skill-integration-worker@sha256:fixed-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=8,
            timeout_seconds=5.0,
        ),
    )


def put_skill_package(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    *,
    instruction: str = "Recompute the final arithmetic before answering.",
    model_fingerprint: str | None = None,
    revision: int = 0,
    parent_package_ref: ArtifactRef | None = None,
    source_kind: SkillSourceKind = SkillSourceKind.FIRST_PARTY,
) -> ArtifactRef:
    package = SkillPackage(
        skill_id="verify-arithmetic",
        revision=revision,
        parent_package_ref=parent_package_ref,
        name="Verify arithmetic",
        summary="Checks arithmetic before returning a final answer.",
        activation_guidance="Use for arithmetic questions.",
        applicability_tags=("arithmetic", "verification"),
        rules=(SkillRule(rule_id="recompute", instruction=instruction),),
        procedure="Solve, recompute independently, then return the answer.",
        examples=(
            SkillExample(
                input="What is 20 + 22?",
                output="42",
                explanation="The independent sum confirms the result.",
            ),
        ),
        compatible_model_fingerprints=(
            spec.model_fingerprint if model_fingerprint is None else model_fingerprint,
        ),
        runtime_fingerprints=(spec.runtime_fingerprint,),
        license=SkillLicense(
            spdx_expression="Apache-2.0",
            source_kind=source_kind,
            provenance_refs=(store.put_bytes(b"skill source", media_type="text/plain"),),
            compliance_review_ref=store.put_json(
                {"approved": True},
                media_type="application/vnd.spiral-harness.compliance-review.v1+json",
            ),
        ),
    )
    return store.put_json(package, media_type=SKILL_PACKAGE_MEDIA_TYPE)


def put_harness(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    *,
    prompt: str = BASE_PROMPT,
    skill_ref: ArtifactRef | None = None,
    extra_component: HarnessComponentRef | None = None,
) -> tuple[ArtifactRef, ArtifactRef]:
    prompt_ref = store.put_bytes(prompt.encode("utf-8"), media_type="text/plain")
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
    if extra_component is not None:
        components.append(extra_component)
    manifest = HarnessManifest(
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        trusted_plane_version="skill-execution-integration-v1",
        components=tuple(components),
    )
    return store.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE), prompt_ref


def paired_schedule(
    prompt_harness_ref: ArtifactRef,
    skill_harness_ref: ArtifactRef,
    *,
    phase: EvaluationPhase = EvaluationPhase.GATE,
) -> EvaluationBatchSchedule:
    return EvaluationBatchSchedule(
        study="skill-execution-integration",
        kind="skill-vs-prompt",
        phase=phase,
        query=0,
        master_seed=20260812,
        parent_harness_id=prompt_harness_ref.sha256,
        candidate_harness_id=skill_harness_ref.sha256,
        task_ids=("arithmetic-1",),
        search_runs=(0,),
        repeat_seeds=(7,),
        max_attempts_per_cell=1,
        token_ceiling_per_attempt=8,
    )


@dataclass(frozen=True)
class PairRun:
    store: ArtifactStore
    spec: FrozenModelSpec
    materializer: HarnessMaterializer
    prompt_harness_ref: ArtifactRef
    skill_harness_ref: ArtifactRef
    prompt_ref: ArtifactRef
    skill_ref: ArtifactRef
    prompt_harness: ResolvedHarness
    skill_harness: ResolvedHarness
    schedule: EvaluationBatchSchedule
    ledger: AttemptLedger
    preflight_ref: ArtifactRef
    records: tuple[ScheduledExecutionRecord, ...]
    backend: ReplayBackend


def execute_pair(
    root: Path,
    *,
    candidate_output: str = "skill-backed: 42",
    phase: EvaluationPhase = EvaluationPhase.GATE,
) -> PairRun:
    store = ArtifactStore(root / "cas")
    spec = fixed_spec()
    skill_ref = put_skill_package(store, spec)
    prompt_harness_ref, _ = put_harness(store, spec)
    skill_harness_ref, prompt_ref = put_harness(store, spec, skill_ref=skill_ref)
    materializer = HarnessMaterializer(store, spec=spec)
    prompt_harness = materializer.materialize(prompt_harness_ref)
    skill_harness = materializer.materialize(skill_harness_ref)
    schedule = paired_schedule(
        prompt_harness_ref,
        skill_harness_ref,
        phase=phase,
    )
    task = CandidateTask(task_id="arithmetic-1", question="What is 20 + 22?")

    requests = {}
    outputs = {
        EvaluationSide.PARENT: "prompt-only: 42",
        EvaluationSide.CANDIDATE: candidate_output,
    }
    for cell in schedule.iter_cells():
        harness = prompt_harness if cell.side is EvaluationSide.PARENT else skill_harness
        request = materialize_request(
            task,
            harness,
            seed=schedule.seed_for(cell, attempt_index=0),
        )
        requests[cell.side] = request

    backend = ReplayBackend(
        fingerprint=BACKEND_FINGERPRINT,
        responses={
            replay_key(spec, request): BackendResponse(
                output=outputs[side],
                usage=BackendTokenUsage(input_tokens=3, output_tokens=1),
            )
            for side, request in requests.items()
        },
    )
    ledger = AttemptLedger(
        store,
        ledger_id="skill-integration-ledger",
        budget=AttemptBudget(
            max_attempts=schedule.required_attempts,
            max_total_tokens=schedule.required_tokens,
            max_tokens_per_attempt=schedule.token_ceiling_per_attempt,
        ),
    )
    preflight_ref = publish_schedule_preflight(
        store,
        preflight_attempt_budget(schedule, ledger, spec),
    )
    runner = FixedModelRunner(spec=spec, backend=backend, attempt_ledger=ledger)
    expected_tail: ArtifactRef | None = None
    previous_receipt_ref: ArtifactRef | None = None
    records = []
    for cell in schedule.iter_cells():
        harness_ref = (
            prompt_harness_ref if cell.side is EvaluationSide.PARENT else skill_harness_ref
        )
        record = execute_scheduled_attempt(
            runner=runner,
            schedule=schedule,
            preflight_ref=preflight_ref,
            expected_previous_ledger_tail_ref=expected_tail,
            previous_receipt_ref=previous_receipt_ref,
            cell=cell,
            attempt_index=0,
            task=task,
            harness_ref=harness_ref,
        )
        records.append(record)
        expected_tail = record.outcome_ref
        previous_receipt_ref = record.receipt_ref

    return PairRun(
        store=store,
        spec=spec,
        materializer=materializer,
        prompt_harness_ref=prompt_harness_ref,
        skill_harness_ref=skill_harness_ref,
        prompt_ref=prompt_ref,
        skill_ref=skill_ref,
        prompt_harness=prompt_harness,
        skill_harness=skill_harness,
        schedule=schedule,
        ledger=ledger,
        preflight_ref=preflight_ref,
        records=tuple(records),
        backend=backend,
    )


def record_for(run: PairRun, side: EvaluationSide) -> ScheduledExecutionRecord:
    return next(record for record in run.records if record.cell.side is side)


def replay_pair(run: PairRun) -> None:
    replay_trusted_usage(
        run.store,
        schedule=run.schedule,
        preflight_ref=run.preflight_ref,
        attempt_ledger=run.ledger,
        receipt_refs=tuple(record.receipt_ref for record in run.records),
    )


def test_skill_materialization_changes_the_exact_request_but_preserves_pairing(
    tmp_path: Path,
) -> None:
    run = execute_pair(tmp_path)
    prompt_record = record_for(run, EvaluationSide.PARENT)
    skill_record = record_for(run, EvaluationSide.CANDIDATE)
    prompt_request = prompt_record.execution.request
    skill_request = skill_record.execution.request

    assert prompt_request.fingerprint != skill_request.fingerprint
    assert replay_key(run.spec, prompt_request) != replay_key(run.spec, skill_request)
    assert prompt_record.execution.execution_fingerprint == (
        skill_record.execution.execution_fingerprint
    )
    assert prompt_record.execution.execution_fingerprint == paired_execution_fingerprint(
        run.spec,
        prompt_record.execution.task,
        seed=prompt_request.seed,
        backend_fingerprint=BACKEND_FINGERPRINT,
    )
    assert prompt_record.execution.output == "prompt-only: 42"
    assert skill_record.execution.output == "skill-backed: 42"
    assert skill_request.skill_disclosure == run.skill_harness.skill_disclosure
    assert skill_request.harness_ref == run.skill_harness_ref
    assert skill_request.skill_disclosure is not None
    assert skill_request.skill_disclosure.level is SkillDisclosureLevel.RULES
    assert set(skill_request.model_dump()) >= {
        "base_system_prompt",
        "skill_disclosure",
        "system_prompt",
    }
    assert not set(skill_request.model_dump()).intersection({"selected", "loaded", "activated"})
    assert set(run.backend.calls) == {
        replay_key(run.spec, prompt_request),
        replay_key(run.spec, skill_request),
    }

    replay_pair(run)
    assert (
        run.materializer.verify_execution_request(
            run.prompt_harness_ref,
            prompt_record.execution,
        )
        == run.prompt_harness
    )
    assert (
        run.materializer.verify_execution_request(
            run.skill_harness_ref,
            skill_record.execution,
        )
        == run.skill_harness
    )


def test_settled_receipts_prove_request_inclusion_not_adherence_or_benefit(
    tmp_path: Path,
) -> None:
    run = execute_pair(
        tmp_path,
        candidate_output="I ignored the disclosed rules and returned 42 directly.",
        phase=EvaluationPhase.PROBE,
    )
    candidate_record = record_for(run, EvaluationSide.CANDIDATE)

    evidence_ref = publish_settled_skill_request_inclusion(
        run.store,
        schedule=run.schedule,
        preflight_ref=run.preflight_ref,
        attempt_ledger=run.ledger,
        receipt_refs=tuple(record.receipt_ref for record in run.records),
        candidate_harness_ref=run.skill_harness_ref,
    )
    evidence = run.store.get_json(
        evidence_ref,
        SettledSkillRequestInclusionEvidence,
    )
    verified = verify_settled_skill_request_inclusion(
        run.store,
        evidence_ref=evidence_ref,
        schedule=run.schedule,
        preflight_ref=run.preflight_ref,
        attempt_ledger=run.ledger,
        candidate_harness_ref=run.skill_harness_ref,
    )

    assert verified == evidence
    assert evidence_ref.media_type == SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE
    assert evidence.claim == SETTLED_SKILL_REQUEST_INCLUSION_CLAIM
    assert evidence.candidate_harness_ref == run.skill_harness_ref
    assert evidence.skill_disclosure == run.skill_harness.skill_disclosure
    assert evidence.skill_disclosure.package_ref == run.skill_ref
    assert evidence.observations[0].receipt_ref == candidate_record.receipt_ref
    assert evidence.observations[0].execution_ref == candidate_record.execution_ref
    assert evidence.observations[0].request_sha256 == candidate_record.execution.request_sha256
    assert candidate_record.execution.output == (
        "I ignored the disclosed rules and returned 42 directly."
    )
    assert not set(evidence.model_dump()).intersection(
        {"activated", "adhered", "behavior_changed", "benefited", "passed"}
    )


def test_immediate_rules_revision_changes_request_and_replay_but_not_pairing(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    spec = fixed_spec()
    before_ref = put_skill_package(
        store,
        spec,
        source_kind=SkillSourceKind.GENERATED,
    )
    after_ref = put_skill_package(
        store,
        spec,
        instruction="Recompute twice using independent arithmetic decompositions.",
        revision=1,
        parent_package_ref=before_ref,
        source_kind=SkillSourceKind.GENERATED,
    )
    loader = SkillPackageLoader(store)
    before, after = loader.verify_revision(
        before_ref=before_ref,
        after_ref=after_ref,
        expected_component_name="verify-arithmetic",
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
    )
    assert before.skill_id == after.skill_id
    assert after.revision == before.revision + 1
    assert before.rules != after.rules

    before_harness_ref, _ = put_harness(store, spec, skill_ref=before_ref)
    after_harness_ref, _ = put_harness(store, spec, skill_ref=after_ref)
    materializer = HarnessMaterializer(store, spec=spec)
    before_harness = materializer.materialize(before_harness_ref)
    after_harness = materializer.materialize(after_harness_ref)
    task = CandidateTask(task_id="arithmetic-1", question="What is 20 + 22?")
    seed = 314159
    before_request = materialize_request(task, before_harness, seed=seed)
    after_request = materialize_request(task, after_harness, seed=seed)

    assert before_request.fingerprint != after_request.fingerprint
    assert replay_key(spec, before_request) != replay_key(spec, after_request)
    before_pair = paired_execution_fingerprint(
        spec,
        task,
        seed=seed,
        backend_fingerprint=BACKEND_FINGERPRINT,
    )
    after_pair = paired_execution_fingerprint(
        spec,
        task,
        seed=seed,
        backend_fingerprint=BACKEND_FINGERPRINT,
    )
    assert before_pair == after_pair


def test_exact_harness_rejoin_rejects_forged_disclosure_package_and_request_swaps(
    tmp_path: Path,
) -> None:
    run = execute_pair(tmp_path)
    replay_pair(run)
    prompt_execution = record_for(run, EvaluationSide.PARENT).execution
    skill_execution = record_for(run, EvaluationSide.CANDIDATE).execution

    with pytest.raises(HarnessMaterializationError, match="exact materialized harness"):
        run.materializer.verify_execution_request(
            run.prompt_harness_ref,
            skill_execution,
        )
    with pytest.raises(HarnessMaterializationError, match="exact materialized harness"):
        run.materializer.verify_execution_request(
            run.skill_harness_ref,
            prompt_execution,
        )

    loader = SkillPackageLoader(run.store)
    metadata_disclosure = loader.disclose(
        run.skill_ref,
        level=SkillDisclosureLevel.METADATA,
        model_fingerprint=run.spec.model_fingerprint,
        runtime_fingerprint=run.spec.runtime_fingerprint,
    )
    forged_harness = ResolvedHarness.from_skill(
        harness_ref=run.skill_harness_ref,
        base_system_prompt=run.skill_harness.base_system_prompt,
        skill_disclosure=metadata_disclosure,
    )
    forged_request = materialize_request(
        skill_execution.task,
        forged_harness,
        seed=skill_execution.seed,
    )
    forged_execution = ModelExecution.model_validate(
        skill_execution.model_copy(
            update={
                "request": forged_request,
                "request_sha256": forged_request.fingerprint,
            }
        ),
        strict=True,
    )
    with pytest.raises(HarnessMaterializationError, match="exact materialized harness"):
        run.materializer.verify_execution_request(
            run.skill_harness_ref,
            forged_execution,
        )

    wrong_skill_ref = put_skill_package(
        run.store,
        run.spec,
        instruction="Trust the first arithmetic result without recomputing.",
    )
    wrong_disclosure = loader.disclose(
        wrong_skill_ref,
        level=SkillDisclosureLevel.RULES,
        model_fingerprint=run.spec.model_fingerprint,
        runtime_fingerprint=run.spec.runtime_fingerprint,
    )
    wrong_package_harness = ResolvedHarness.from_skill(
        harness_ref=run.skill_harness_ref,
        base_system_prompt=run.skill_harness.base_system_prompt,
        skill_disclosure=wrong_disclosure,
    )
    wrong_request = materialize_request(
        skill_execution.task,
        wrong_package_harness,
        seed=skill_execution.seed,
    )
    wrong_execution = ModelExecution.model_validate(
        skill_execution.model_copy(
            update={
                "request": wrong_request,
                "request_sha256": wrong_request.fingerprint,
            }
        ),
        strict=True,
    )
    with pytest.raises(HarnessMaterializationError, match="exact materialized harness"):
        run.materializer.verify_execution_request(
            run.skill_harness_ref,
            wrong_execution,
        )

    superseding_harness_ref, _ = put_harness(
        run.store,
        run.spec,
        prompt="A later prompt revision.",
        skill_ref=run.skill_ref,
    )
    with pytest.raises(HarnessMaterializationError, match="exact materialized harness"):
        run.materializer.verify_execution_request(
            superseding_harness_ref,
            skill_execution,
        )


@pytest.mark.parametrize("target", ["manifest", "prompt", "package"])
def test_cas_tampering_fails_at_the_receipt_or_exact_harness_rejoin_boundary(
    tmp_path: Path,
    target: str,
) -> None:
    run = execute_pair(tmp_path / target)
    skill_execution = record_for(run, EvaluationSide.CANDIDATE).execution
    tampered_ref = {
        "manifest": run.skill_harness_ref,
        "prompt": run.prompt_ref,
        "package": run.skill_ref,
    }[target]
    run.store.path_for(tampered_ref).write_bytes(b"x" * tampered_ref.size)

    with pytest.raises(
        ExecutionReceiptIntegrityError,
        match="harness request replay failed",
    ):
        replay_pair(run)

    with pytest.raises(HarnessMaterializationError, match="could not be verified"):
        run.materializer.verify_execution_request(
            run.skill_harness_ref,
            skill_execution,
        )


@pytest.mark.parametrize("case", ["unsupported", "incompatible", "tampered-prompt"])
def test_invalid_harness_materialization_fails_before_reservation(
    tmp_path: Path,
    case: str,
) -> None:
    store = ArtifactStore(tmp_path / case)
    spec = fixed_spec()
    skill_ref = None
    extra_component = None
    if case == "unsupported":
        extra_component = HarnessComponentRef(
            name="memory",
            kind=ComponentKind.MEMORY,
            artifact=store.put_bytes(b"untrusted memory", media_type="text/plain"),
        )
    elif case == "incompatible":
        skill_ref = put_skill_package(
            store,
            spec,
            model_fingerprint="foreign-model-fingerprint",
        )
    else:
        skill_ref = put_skill_package(store, spec)
    harness_ref, prompt_ref = put_harness(
        store,
        spec,
        skill_ref=skill_ref,
        extra_component=extra_component,
    )
    if case == "tampered-prompt":
        store.path_for(prompt_ref).write_bytes(b"x" * prompt_ref.size)

    ledger = AttemptLedger(
        store,
        ledger_id=f"pre-reservation-{case}",
        budget=AttemptBudget(
            max_attempts=1,
            max_total_tokens=8,
            max_tokens_per_attempt=8,
        ),
    )
    backend = ReplayBackend(
        fingerprint=BACKEND_FINGERPRINT,
        default_response=BackendResponse(
            output="must not run",
            usage=BackendTokenUsage(input_tokens=1, output_tokens=1),
        ),
    )
    FixedModelRunner(spec=spec, backend=backend, attempt_ledger=ledger)

    with pytest.raises(HarnessMaterializationError):
        HarnessMaterializer(store, spec=spec).materialize(harness_ref)

    state = ledger.state()
    assert state.attempts_used == 0
    assert state.pending_reservation_ref is None
    assert state.tail_ref is None
    assert backend.calls == ()
