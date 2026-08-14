from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from spiral_harness.benchmark._harness_fault_cases import (
    PARTITION_OPENING_MEDIA_TYPE,
    PARTITION_ROSTER_MEDIA_TYPE,
    FaultFamily,
    HarnessFaultAuthorityError,
    PartitionEvaluationGrant,
    PartitionOpening,
    RepairRuleId,
    RuntimeBranch,
    candidate_rule_ids,
    verify_partition_opening,
)
from spiral_harness.benchmark.harness_fault import (
    HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE,
    HARNESS_FAULT_GRADED_OUTCOME_MEDIA_TYPE,
    HarnessFaultComparisonKind,
    HarnessFaultDataError,
    HarnessFaultExplorationGrader,
    HarnessFaultGateEvaluator,
    HarnessFaultGradedBatch,
    HarnessFaultGradedOutcome,
    HarnessFaultGradingError,
    HarnessFaultScheduleClosure,
    HarnessFaultSealedEvaluator,
    parse_harness_fault_output,
)
from spiral_harness.benchmark.harness_fault_authority import HarnessFaultScenarioAuthority
from spiral_harness.benchmark.harness_fault_compiler import (
    HARNESS_FAULT_COMPILATION_MEDIA_TYPE,
    FaultRepairAction,
    HarnessFaultCompilationError,
    HarnessRole,
    compile_fault_repair,
    parse_fault_repair_action,
    verify_fault_compilation,
)
from spiral_harness.benchmark.harness_fault_mechanism import (
    HarnessFaultMechanismError,
    HarnessFaultMechanismRecord,
    HarnessFaultMechanismVerification,
    verify_harness_fault_mechanism,
)
from spiral_harness.benchmark.harness_fault_runtime import (
    RUNTIME_EVENT_MEDIA_TYPE,
    AttestedRuntimeBranchEvent,
    HarnessFaultMiddlewareBackend,
    TrustedRuntimeEventService,
)
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import ArtifactRef, HarnessManifest
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    ATTEMPT_RESERVATION_MEDIA_TYPE,
    AttemptBudget,
    AttemptOutcome,
    AttemptReservation,
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
)
from spiral_harness.execution.materialization import HarnessMaterializer
from spiral_harness.execution.model import (
    FixedModelRunner,
    ReplayBackend,
    materialize_request,
    replay_key,
)
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceipt,
    execute_scheduled_attempt,
)
from spiral_harness.execution.schedule import (
    EvaluationBatchSchedule,
    EvaluationPhase,
    SchedulePreflightCertificate,
    preflight_attempt_budget,
    publish_schedule_preflight,
)
from spiral_harness.storage.artifact_store import ArtifactStore

_EXPLORATION_SALT = b"exploration-authority-salt-v4-0001"
_GATE_SALT = b"gate-authority-salt-v4-0000000001"
_SEALED_SALT = b"sealed-authority-salt-v4-00000001"
_BACKEND_FINGERPRINT = "harness-fault-base-replay@sha256:v4"


def _authority(store: ArtifactStore, *, suffix: bytes = b"") -> HarnessFaultScenarioAuthority:
    return HarnessFaultScenarioAuthority(
        store,
        exploration_salt=_EXPLORATION_SALT + suffix,
        gate_salt=_GATE_SALT + suffix,
        sealed_salt=_SEALED_SALT + suffix,
    )


def _spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="deterministic-replay-with-trusted-middleware",
        backend_fingerprint=_BACKEND_FINGERPRINT,
        model="fixture/harness-fault-base-model",
        revision="snapshot-2026-08-14",
        tokenizer="fixture/frozen-tokenizer",
        tokenizer_revision="snapshot-2026-08-14",
        runtime="python-3.12/harness-fault-multi-surface-v4",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=64,
            timeout_seconds=5.0,
        ),
    )


def test_public_search_view_contains_only_exploration_tasks_and_no_gold_or_openings(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    authority = _authority(store)
    search = authority.public_search_view
    gate = verify_partition_opening(store, authority.issue_gate_grant())
    sealed = verify_partition_opening(store, authority.issue_sealed_grant())
    serialized = search.model_dump_json()

    assert {task.task_id for task in search.exploration_tasks}.isdisjoint(
        {item.task.task_id for item in (*gate.scenarios, *sealed.scenarios)}
    )
    for forbidden in (
        "opening_ref",
        "roster_ref",
        "salt_hex",
        "expected_answer",
        "expected_observable",
        "scenario_commitment",
        _GATE_SALT.decode("ascii"),
        _SEALED_SALT.decode("ascii"),
    ):
        assert forbidden not in serialized
    assert set(type(search.exploration_tasks[0]).model_fields) == {
        "schema_version",
        "task_id",
        "question",
    }
    assert "activation" not in search.exploration_tasks[0].question.casefold()
    assert "benchmark" not in search.exploration_tasks[0].question.casefold()


def test_authority_releases_distinct_phase_evaluators_without_public_hidden_replay_api(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    authority = _authority(store)
    service = TrustedRuntimeEventService()
    capability = service.verification_capability
    exploration = HarnessFaultExplorationGrader(
        store, authority.issue_exploration_grant(), capability
    )
    gate = HarnessFaultGateEvaluator(store, authority.issue_gate_grant(), capability)
    sealed = HarnessFaultSealedEvaluator(store, authority.issue_sealed_grant(), capability)

    assert len(exploration.task_roster()) == len(gate.task_roster()) == len(sealed.task_roster())
    assert set(exploration.task_roster()).isdisjoint(gate.task_roster())
    assert set(gate.task_roster()).isdisjoint(sealed.task_roster())
    for evaluator in (exploration, gate, sealed):
        assert not hasattr(evaluator, "scenario_for_trusted_replay")
        assert not hasattr(evaluator, "hidden_scenarios")
    with pytest.raises(HarnessFaultDataError, match="another evaluator partition"):
        HarnessFaultGateEvaluator(store, authority.issue_sealed_grant(), capability)


def test_partition_opening_recomputes_every_root_and_rejects_cross_authority_or_tamper(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    first = _authority(store)
    second = _authority(store, suffix=b"-independent")
    first_grant = first.issue_gate_grant()
    verified = verify_partition_opening(store, first_grant)

    assert (
        verified.roster.root
        == first_grant.public_commitment.partition(ProtocolPartition.GATE).roster_root
    )
    assert tuple(item.task for item in verified.scenarios) == verified.roster.tasks

    cross_authority = PartitionEvaluationGrant(
        public_commitment=second.public_search_view.public_commitment,
        partition=ProtocolPartition.GATE,
        opening_ref=first_grant.opening_ref,
        roster_ref=first_grant.roster_ref,
    )
    with pytest.raises(HarnessFaultAuthorityError, match="another authority"):
        verify_partition_opening(store, cross_authority)

    opening = store.get_json(first_grant.opening_ref, PartitionOpening)
    changed_first_byte = "00" + opening.salt_hex[2:]
    if changed_first_byte == opening.salt_hex:
        changed_first_byte = "ff" + opening.salt_hex[2:]
    forged_opening_ref = store.put_json(
        opening.model_copy(update={"salt_hex": changed_first_byte}),
        media_type=PARTITION_OPENING_MEDIA_TYPE,
    )
    forged = first_grant.model_copy(update={"opening_ref": forged_opening_ref})
    with pytest.raises(HarnessFaultAuthorityError, match="salt does not match"):
        verify_partition_opening(store, forged)

    foreign_roster = second.issue_gate_grant().roster_ref
    assert foreign_roster.media_type == PARTITION_ROSTER_MEDIA_TYPE
    with pytest.raises(HarnessFaultAuthorityError, match="another authority"):
        verify_partition_opening(
            store,
            first_grant.model_copy(update={"roster_ref": foreign_roster}),
        )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        '{"rule_id":"unknown"}',
        '{"rule_id":"c-50d8b731"}',
        '{"rule_id":"r-13f0a9c2","prompt":"inject"}',
        '{"rule_id":"r-13f0a9c2","rule_id":"r-7bd91e40"}',
    ],
)
def test_proposal_parser_accepts_only_one_opaque_catalog_id(text: str) -> None:
    with pytest.raises(HarnessFaultCompilationError):
        parse_fault_repair_action(text)


def test_compiler_verifier_rebuilds_prompts_parent_graph_and_matched_controls(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    service = TrustedRuntimeEventService()
    spec = _spec()
    opaque_id = candidate_rule_ids()[0]
    action = parse_fault_repair_action(json.dumps({"rule_id": opaque_id.value}))
    compiled = compile_fault_repair(
        store,
        spec,
        action,
        runtime_producer_id=service.producer_id,
    )
    verified = verify_fault_compilation(
        store,
        spec,
        compiled.manifest_ref,
        expected_runtime_producer_id=service.producer_id,
    )
    parent = verified.entry(HarnessRole.FAULTY_PARENT)
    oracle = verified.entry(HarnessRole.ORACLE)
    candidate = verified.entry(HarnessRole.CANDIDATE)
    revert = verified.entry(HarnessRole.REVERT)
    placebo = verified.entry(HarnessRole.PLACEBO)

    assert revert.prompt_ref == parent.prompt_ref
    assert store.get_bytes(revert.prompt_ref) == store.get_bytes(parent.prompt_ref)
    assert candidate.prompt_ref.size == placebo.prompt_ref.size == verified.matched_prompt_bytes
    assert store.get_bytes(candidate.prompt_ref).startswith(b"HFB4_FROZEN_COMPONENT")
    assert store.get_bytes(placebo.prompt_ref).startswith(b"HFB4_FROZEN_COMPONENT")
    revert_manifest = store.get_json(revert.harness_ref, HarnessManifest)
    oracle_manifest = store.get_json(oracle.harness_ref, HarnessManifest)
    assert oracle_manifest.parent is None
    assert revert_manifest.parent == candidate.harness_ref

    first_match = verified.matched_component_bytes[0]
    forged_manifest = verified.model_copy(
        update={
            "matched_component_bytes": (
                first_match.model_copy(update={"byte_count": first_match.byte_count + 1}),
                *verified.matched_component_bytes[1:],
            )
        }
    )
    forged_ref = store.put_json(
        forged_manifest,
        media_type=HARNESS_FAULT_COMPILATION_MEDIA_TYPE,
    )
    with pytest.raises(HarnessFaultCompilationError, match="frozen reconstruction"):
        verify_fault_compilation(
            store,
            spec,
            forged_ref,
            expected_runtime_producer_id=service.producer_id,
        )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "{}",
        '{"answer":"MATCH"}',
        '{"answer":"MATCH","observable":"x","activation":"yes"}',
        '{"answer":"MATCH","answer":"DIFFERENT","observable":"x"}',
        '{"answer":1,"observable":"x"}',
        '{"answer":"OTHER","observable":"x"}',
        '{"answer":"MATCH","observable":" x"}',
        '{"answer":"MATCH","observable":NaN}',
    ],
)
def test_worker_output_has_only_answer_and_observable(text: str) -> None:
    with pytest.raises(HarnessFaultGradingError):
        parse_harness_fault_output(text)


@dataclass(frozen=True, slots=True)
class _LoopResult:
    store: ArtifactStore
    evaluator: HarnessFaultExplorationGrader
    compilation_ref: ArtifactRef
    batch_refs: tuple[ArtifactRef, ...]
    mechanism: HarnessFaultMechanismRecord
    backend: HarnessFaultMiddlewareBackend
    ledgers: tuple[AttemptLedger, ...]


def _schedules(evaluator, compilation) -> tuple[EvaluationBatchSchedule, ...]:
    tasks = evaluator.task_roster()
    candidate_id = compilation.entry(HarnessRole.CANDIDATE).harness_ref.sha256
    controls = (
        (HarnessFaultComparisonKind.MAIN, EvaluationPhase.EXPLORATION, HarnessRole.FAULTY_PARENT),
        (HarnessFaultComparisonKind.REVERT, EvaluationPhase.PROBE, HarnessRole.REVERT),
        (HarnessFaultComparisonKind.PLACEBO, EvaluationPhase.PROBE, HarnessRole.PLACEBO),
    )
    return tuple(
        EvaluationBatchSchedule(
            study="harness-fault-v4-multi-surface-replay",
            kind=f"mechanism-{kind.value}",
            phase=phase,
            query=index,
            master_seed=20270814,
            parent_harness_id=compilation.entry(role).harness_ref.sha256,
            candidate_harness_id=candidate_id,
            task_ids=tasks,
            search_runs=(0,),
            repeat_seeds=(0,),
            max_attempts_per_cell=1,
            token_ceiling_per_attempt=64,
        )
        for index, (kind, phase, role) in enumerate(controls)
    )


def _base_replay_responses(
    store: ArtifactStore,
    evaluator: HarnessFaultExplorationGrader,
    spec: FrozenModelSpec,
    compilation,
    schedules: tuple[EvaluationBatchSchedule, ...],
) -> dict[str, BackendResponse]:
    materializer = HarnessMaterializer(store, spec=spec)
    refs = {entry.harness_ref.sha256: entry.harness_ref for entry in compilation.entries}
    responses = {}
    for schedule in schedules:
        for cell in schedule.iter_cells():
            harness_ref = refs[schedule.harness_id_for(cell.side)]
            request = materialize_request(
                evaluator.load_task(cell.task_id),
                materializer.materialize(harness_ref),
                seed=schedule.seed_for(cell),
            )
            responses[replay_key(spec, request)] = BackendResponse(
                output="BASE_MODEL_ACK",
                usage=BackendTokenUsage(input_tokens=2, output_tokens=2),
                cost_usd=0.0,
            )
    return responses


def _run_rule(root, catalog_index: int) -> _LoopResult:
    store = ArtifactStore(root)
    authority = _authority(store)
    event_service = TrustedRuntimeEventService()
    evaluator = HarnessFaultExplorationGrader(
        store,
        authority.issue_exploration_grant(),
        event_service.verification_capability,
    )
    spec = _spec()
    opaque_id = candidate_rule_ids()[catalog_index]
    action = parse_fault_repair_action(json.dumps({"rule_id": opaque_id.value}))
    compiled = compile_fault_repair(
        store,
        spec,
        action,
        runtime_producer_id=event_service.producer_id,
    )
    schedules = _schedules(evaluator, compiled.manifest)
    base_backend = ReplayBackend(
        fingerprint=_BACKEND_FINGERPRINT,
        responses=_base_replay_responses(
            store,
            evaluator,
            spec,
            compiled.manifest,
            schedules,
        ),
    )
    runtime_backend = HarnessFaultMiddlewareBackend(
        store=store,
        spec=spec,
        underlying=base_backend,
        compilation_ref=compiled.manifest_ref,
        event_service=event_service,
    )
    refs = {entry.harness_ref.sha256: entry.harness_ref for entry in compiled.manifest.entries}
    batch_refs = []
    ledgers = []
    for batch_index, (kind, schedule) in enumerate(
        zip(HarnessFaultComparisonKind, schedules, strict=True)
    ):
        # Final verification requires a frozen live writer per batch.  The one
        # spare slot is used by the fail-closed ledger-advance regression.
        capacity = schedule.cell_count + 1
        ledger = AttemptLedger(
            store,
            ledger_id=f"harness-fault-v4-rule-{catalog_index}-batch-{batch_index}",
            budget=AttemptBudget(
                max_attempts=capacity,
                max_total_tokens=capacity * 64,
                max_tokens_per_attempt=64,
            ),
        )
        runner = FixedModelRunner(spec=spec, backend=runtime_backend, attempt_ledger=ledger)
        schedule_ref = evaluator.publish_schedule(
            schedule,
            comparison_kind=kind,
            compilation_ref=compiled.manifest_ref,
            spec=spec,
        )
        preflight = preflight_attempt_budget(schedule, ledger, spec)
        preflight_ref = publish_schedule_preflight(store, preflight)
        tail = preflight.ledger_tail_ref
        previous_receipt = None
        receipt_refs = []
        for cell in schedule.iter_cells():
            record = execute_scheduled_attempt(
                runner=runner,
                schedule=schedule,
                preflight_ref=preflight_ref,
                expected_previous_ledger_tail_ref=tail,
                previous_receipt_ref=previous_receipt,
                cell=cell,
                attempt_index=0,
                task=evaluator.load_task(cell.task_id),
                harness_ref=refs[schedule.harness_id_for(cell.side)],
            )
            receipt_refs.append(record.receipt_ref)
            tail = record.outcome_ref
            previous_receipt = record.receipt_ref
        graded = evaluator.grade_receipt_batch(
            schedule_ref=schedule_ref,
            preflight_ref=preflight_ref,
            attempt_ledger=ledger,
            receipt_refs=receipt_refs,
            runtime_backend=runtime_backend,
        )
        batch_refs.append(graded.batch_ref)
        ledgers.append(ledger)
    mechanism = verify_harness_fault_mechanism(
        store,
        evaluator,
        compilation_ref=compiled.manifest_ref,
        batch_refs=batch_refs,
        attempt_ledgers=ledgers,
    )
    return _LoopResult(
        store=store,
        evaluator=evaluator,
        compilation_ref=compiled.manifest_ref,
        batch_refs=tuple(batch_refs),
        mechanism=mechanism,
        backend=runtime_backend,
        ledgers=tuple(ledgers),
    )


@pytest.fixture(scope="module")
def routed_result(tmp_path_factory) -> _LoopResult:
    root = tmp_path_factory.mktemp("harness-fault-v4-routed")
    return _run_rule(root / "routed-policy", candidate_rule_ids().index(RepairRuleId.ROUTED_POLICY))


def test_catalog_exposes_multi_surface_anti_constant_envelope(
    routed_result: _LoopResult,
) -> None:
    item = routed_result.mechanism.verification
    assert item.mechanism_passed
    assert item.candidate_behavior_rate == item.oracle_behavior_rate == 0.75
    assert item.best_constant_branch_selector_rate == 11 / 28
    assert item.best_single_family_patch_rate == 12 / 28
    assert item.best_constant_output_rate == 1 / 28
    assert item.oracle_minus_best_constant_branch == 10 / 28
    assert item.oracle_minus_best_single_family_patch == 9 / 28
    assert item.evidence_scope == "deterministic-multi-surface-trust-closure-slice"
    assert item.base_model_output_drives_behavior is False
    assert item.supports_optimizer_capability_claim is False
    assert item.supports_llm_solver_claim is False
    assert item.supports_self_evolution_claim is False
    assert item.supports_live_model_benchmark_claim is False


def test_runtime_branch_is_signed_actual_middleware_event_and_repeats_are_paired(
    routed_result: _LoopResult,
) -> None:
    result = routed_result
    verification = result.mechanism.verification

    assert verification.family_count == 7
    assert verification.surface_count == 6
    assert verification.scenario_count == 28
    assert verification.group_count == 7
    assert verification.main_candidate_event_count == 28
    assert verification.repairable_event_count == 7
    assert verification.null_event_count == 7
    assert verification.unrepairable_event_count == 7
    assert verification.shift_hard_negative_event_count == 7
    assert verification.repairable_behavior_rate == 1.0
    assert verification.null_behavior_rate == 1.0
    assert verification.unrepairable_behavior_rate == 0.0
    assert verification.shift_hard_negative_behavior_rate == 1.0
    assert verification.parent_pair_count == 28
    assert verification.revert_pair_count == 28
    assert verification.placebo_pair_count == 28
    assert verification.mean_paired_gain_vs_parent == 11 / 28
    assert verification.mean_paired_gain_vs_revert == 11 / 28
    assert verification.mean_paired_gain_vs_placebo == 11 / 28
    assert "mechanism_passed" not in type(verification).model_fields
    assert "activation_passed" not in type(verification).model_fields

    batch = result.store.get_json(result.batch_refs[0], HarnessFaultGradedBatch)
    outcome = result.store.get_json(batch.outcome_refs[0], HarnessFaultGradedOutcome)
    event = result.store.get_json(outcome.runtime_event_ref, AttestedRuntimeBranchEvent)
    closure = result.store.get_json(batch.schedule_ref, HarnessFaultScheduleClosure)
    assert event.branch in set(RuntimeBranch)
    assert event.base_output_sha256 != event.final_output_sha256
    assert outcome.observation.seed == closure.schedule.seed_for(outcome.cell)


def test_mechanism_rejects_cherry_picked_batch_even_with_real_outcomes(
    routed_result: _LoopResult,
) -> None:
    result = routed_result
    main = result.store.get_json(result.batch_refs[0], HarnessFaultGradedBatch)
    forged = main.model_copy(update={"outcome_refs": main.outcome_refs[:-2]})
    forged_ref = result.store.put_json(
        forged,
        media_type=HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE,
    )

    with pytest.raises(HarnessFaultMechanismError, match="exact evaluator roster/schedule"):
        verify_harness_fault_mechanism(
            result.store,
            result.evaluator,
            compilation_ref=result.compilation_ref,
            batch_refs=(forged_ref, *result.batch_refs[1:]),
            attempt_ledgers=result.ledgers,
        )


def test_mechanism_rejects_tampered_signed_branch_event_inside_complete_batch(
    routed_result: _LoopResult,
) -> None:
    result = routed_result
    main = result.store.get_json(result.batch_refs[0], HarnessFaultGradedBatch)
    outcome = result.store.get_json(main.outcome_refs[0], HarnessFaultGradedOutcome)
    event = result.store.get_json(outcome.runtime_event_ref, AttestedRuntimeBranchEvent)
    other_branch = (
        RuntimeBranch.LEGACY if event.branch is RuntimeBranch.SAFE else RuntimeBranch.SAFE
    )
    forged_event_ref = result.store.put_json(
        event.model_copy(update={"branch": other_branch}),
        media_type=RUNTIME_EVENT_MEDIA_TYPE,
    )
    forged_outcome_ref = result.store.put_json(
        outcome.model_copy(update={"runtime_event_ref": forged_event_ref}),
        media_type=HARNESS_FAULT_GRADED_OUTCOME_MEDIA_TYPE,
    )
    forged_outcomes = (forged_outcome_ref, *main.outcome_refs[1:])
    forged_batch_ref = result.store.put_json(
        main.model_copy(update={"outcome_refs": forged_outcomes}),
        media_type=HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE,
    )

    with pytest.raises(ValueError, match="attestation is invalid"):
        verify_harness_fault_mechanism(
            result.store,
            result.evaluator,
            compilation_ref=result.compilation_ref,
            batch_refs=(forged_batch_ref, *result.batch_refs[1:]),
            attempt_ledgers=result.ledgers,
        )


def _replace_main_batch(
    result: _LoopResult,
    batch: HarnessFaultGradedBatch,
) -> tuple[ArtifactRef, ...]:
    replacement = result.store.put_json(
        batch,
        media_type=HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE,
    )
    return (replacement, *result.batch_refs[1:])


@pytest.mark.parametrize("tail_kind", ["nonexistent", "foreign", "forged"])
def test_final_verifier_replays_and_rejects_untrusted_usage_tail(
    routed_result: _LoopResult,
    tail_kind: str,
) -> None:
    result = routed_result
    main = result.store.get_json(result.batch_refs[0], HarnessFaultGradedBatch)
    if tail_kind == "nonexistent":
        tail = ArtifactRef(
            sha256="f" * 64,
            size=1,
            media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
        )
    elif tail_kind == "foreign":
        other = result.store.get_json(result.batch_refs[1], HarnessFaultGradedBatch)
        tail = other.usage.ledger_tail_refs[0]
    else:
        original = result.store.get_json(main.usage.ledger_tail_refs[0], AttemptOutcome)
        tail = result.store.put_json(
            original.model_copy(update={"writer_epoch_id": "e" * 64}),
            media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
        )
    forged = main.model_copy(
        update={"usage": main.usage.model_copy(update={"ledger_tail_refs": (tail,)})}
    )

    with pytest.raises(HarnessFaultMechanismError, match="trusted usage"):
        verify_harness_fault_mechanism(
            result.store,
            result.evaluator,
            compilation_ref=result.compilation_ref,
            batch_refs=_replace_main_batch(result, forged),
            attempt_ledgers=result.ledgers,
        )


def test_final_verifier_rejects_forged_usage_totals(
    routed_result: _LoopResult,
) -> None:
    result = routed_result
    main = result.store.get_json(result.batch_refs[0], HarnessFaultGradedBatch)
    forged_usage = main.usage.model_copy(
        update={
            "reported_tokens": main.usage.reported_tokens + 1,
            "charged_tokens": main.usage.charged_tokens + 1,
        }
    )

    with pytest.raises(HarnessFaultMechanismError, match="trusted usage"):
        verify_harness_fault_mechanism(
            result.store,
            result.evaluator,
            compilation_ref=result.compilation_ref,
            batch_refs=_replace_main_batch(
                result,
                main.model_copy(update={"usage": forged_usage}),
            ),
            attempt_ledgers=result.ledgers,
        )


def test_final_verifier_rejects_receipt_with_wrong_preflight(
    routed_result: _LoopResult,
) -> None:
    result = routed_result
    main = result.store.get_json(result.batch_refs[0], HarnessFaultGradedBatch)
    other = result.store.get_json(result.batch_refs[1], HarnessFaultGradedBatch)
    receipt = result.store.get_json(main.usage.receipt_refs[0], ExecutionReceipt)
    other_receipt = result.store.get_json(other.usage.receipt_refs[0], ExecutionReceipt)
    other_preflight = result.store.get_json(
        other_receipt.preflight_ref,
        SchedulePreflightCertificate,
    )
    forged_receipt_ref = result.store.put_json(
        receipt.model_copy(
            update={
                "preflight_ref": other_receipt.preflight_ref,
                "preflight_fingerprint": other_preflight.fingerprint,
            }
        ),
        media_type=EXECUTION_RECEIPT_MEDIA_TYPE,
    )
    forged_receipts = (forged_receipt_ref, *main.usage.receipt_refs[1:])
    forged_usage = main.usage.model_copy(update={"receipt_refs": forged_receipts})

    with pytest.raises(HarnessFaultMechanismError, match="trusted usage"):
        verify_harness_fault_mechanism(
            result.store,
            result.evaluator,
            compilation_ref=result.compilation_ref,
            batch_refs=_replace_main_batch(
                result,
                main.model_copy(update={"usage": forged_usage}),
            ),
            attempt_ledgers=result.ledgers,
        )


def test_final_verifier_rejects_disconnected_reservation_chain(
    routed_result: _LoopResult,
) -> None:
    result = routed_result
    main = result.store.get_json(result.batch_refs[0], HarnessFaultGradedBatch)
    receipts = [result.store.get_json(ref, ExecutionReceipt) for ref in main.usage.receipt_refs]
    selected_index, receipt, reservation = next(
        (index, receipt, reservation)
        for index, receipt in enumerate(receipts)
        if (
            reservation := result.store.get_json(
                receipt.reservation_ref,
                AttemptReservation,
            )
        ).sequence
        > 0
    )
    disconnected_ref = ArtifactRef(
        sha256="d" * 64,
        size=1,
        media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
    )
    forged_reservation_ref = result.store.put_json(
        reservation.model_copy(update={"previous_outcome_ref": disconnected_ref}),
        media_type=ATTEMPT_RESERVATION_MEDIA_TYPE,
    )
    outcome = result.store.get_json(receipt.outcome_ref, AttemptOutcome)
    forged_outcome_ref = result.store.put_json(
        outcome.model_copy(update={"reservation_ref": forged_reservation_ref}),
        media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
    )
    forged_receipt_ref = result.store.put_json(
        receipt.model_copy(
            update={
                "reservation_ref": forged_reservation_ref,
                "outcome_ref": forged_outcome_ref,
                "ledger_tail_ref": forged_outcome_ref,
            }
        ),
        media_type=EXECUTION_RECEIPT_MEDIA_TYPE,
    )
    receipt_refs = list(main.usage.receipt_refs)
    receipt_refs[selected_index] = forged_receipt_ref
    forged_usage = main.usage.model_copy(update={"receipt_refs": tuple(receipt_refs)})

    with pytest.raises(HarnessFaultMechanismError, match="trusted usage"):
        verify_harness_fault_mechanism(
            result.store,
            result.evaluator,
            compilation_ref=result.compilation_ref,
            batch_refs=_replace_main_batch(
                result,
                main.model_copy(update={"usage": forged_usage}),
            ),
            attempt_ledgers=result.ledgers,
        )


def test_final_verifier_rejects_foreign_live_ledger(
    routed_result: _LoopResult,
) -> None:
    result = routed_result
    foreign = AttemptLedger(
        result.store,
        ledger_id="harness-fault-v4-foreign-ledger",
        budget=result.ledgers[0].budget,
    )

    with pytest.raises(HarnessFaultMechanismError, match="live ledger"):
        verify_harness_fault_mechanism(
            result.store,
            result.evaluator,
            compilation_ref=result.compilation_ref,
            batch_refs=result.batch_refs,
            attempt_ledgers=(foreign, *result.ledgers[1:]),
        )


def test_final_verifier_rejects_ledger_advance_after_grading(tmp_path) -> None:
    result = _run_rule(
        tmp_path / "advanced",
        candidate_rule_ids().index(RepairRuleId.ROUTED_POLICY),
    )
    ledger = result.ledgers[0]
    reservation_ref = ledger.reserve(
        task_fingerprint="1" * 64,
        execution_fingerprint="2" * 64,
        request_sha256="3" * 64,
        token_ceiling=1,
    )
    ledger.burn(reservation_ref, error_class="post-grade-ledger-advance")

    with pytest.raises(HarnessFaultMechanismError, match="trusted usage"):
        verify_harness_fault_mechanism(
            result.store,
            result.evaluator,
            compilation_ref=result.compilation_ref,
            batch_refs=result.batch_refs,
            attempt_ledgers=result.ledgers,
        )


def test_schedule_rejects_task_subset_before_any_execution(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    authority = _authority(store)
    service = TrustedRuntimeEventService()
    evaluator = HarnessFaultExplorationGrader(
        store, authority.issue_exploration_grant(), service.verification_capability
    )
    spec = _spec()
    action = FaultRepairAction(rule_id=candidate_rule_ids()[0])
    compiled = compile_fault_repair(
        store,
        spec,
        action,
        runtime_producer_id=service.producer_id,
    )
    complete = _schedules(evaluator, compiled.manifest)[0]
    subset = complete.model_copy(update={"task_ids": complete.task_ids[:-1]})

    with pytest.raises(HarnessFaultGradingError, match="complete partition roster"):
        evaluator.publish_schedule(
            subset,
            comparison_kind=HarnessFaultComparisonKind.MAIN,
            compilation_ref=compiled.manifest_ref,
            spec=spec,
        )


def test_scope_is_explicitly_multi_surface_but_capability_claims_remain_closed() -> None:
    assert len(FaultFamily) == 7
    assert len(candidate_rule_ids()) == 6
    assert (
        "multi-surface"
        in HarnessFaultMechanismVerification.model_fields["verifier_version"].default
    )
    assert (
        HarnessFaultMechanismVerification.model_fields[
            "supports_optimizer_capability_claim"
        ].default
        is False
    )
