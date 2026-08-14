"""Full-batch v4 closure over multi-surface branches and paired cells."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, model_validator

from spiral_harness.benchmark._harness_fault_cases import (
    FaultFamily,
    FaultSurface,
    HiddenScenarioSpec,
    RepairRuleId,
    RuntimeBranch,
    ScenarioRole,
    anti_cheat_summary,
    evaluate_branch,
    verify_partition_opening,
)
from spiral_harness.benchmark.harness_fault import (
    HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE,
    HARNESS_FAULT_GRADED_OUTCOME_MEDIA_TYPE,
    HARNESS_FAULT_SCHEDULE_MEDIA_TYPE,
    HarnessFaultComparisonKind,
    HarnessFaultExplorationGrader,
    HarnessFaultGradedBatch,
    HarnessFaultGradedOutcome,
    HarnessFaultScheduleClosure,
    parse_harness_fault_output,
)
from spiral_harness.benchmark.harness_fault_compiler import (
    HARNESS_FAULT_COMPILATION_MEDIA_TYPE,
    HarnessFaultCompilationManifest,
    HarnessRole,
    verify_fault_compilation,
)
from spiral_harness.benchmark.harness_fault_errors import HarnessFaultMechanismError
from spiral_harness.benchmark.harness_fault_ledger import (
    LedgerKey,
    index_exact_live_ledgers,
    replay_exact_batch_usage,
    reverify_live_ledger_tails,
)
from spiral_harness.benchmark.harness_fault_runtime import (
    RUNTIME_EVENT_MEDIA_TYPE,
    AttestedRuntimeBranchEvent,
)
from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptDisposition,
    AttemptOutcome,
    CandidateTask,
    ExecutionStatus,
    ModelExecution,
)
from spiral_harness.execution.materialization import HarnessMaterializer
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceipt,
)
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationCellKey,
    EvaluationSide,
    SchedulePreflightCertificate,
)
from spiral_harness.storage.artifact_store import ArtifactStore

HARNESS_FAULT_MECHANISM_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-mechanism-verification.v4+json"
)
MECHANISM_VERIFIER_VERSION = "spiral-harness.harness-fault-mechanism:v4-multi-surface"


class HarnessFaultMechanismVerification(ImmutableModel):
    """Primitive trust-closure statistics, never an optimizer-capability result."""

    schema_version: Literal["4"] = "4"
    verifier_version: Literal["spiral-harness.harness-fault-mechanism:v4-multi-surface"] = (
        MECHANISM_VERIFIER_VERSION
    )
    evidence_scope: Literal["deterministic-multi-surface-trust-closure-slice"] = (
        "deterministic-multi-surface-trust-closure-slice"
    )
    base_model_output_drives_behavior: Literal[False] = False
    supports_optimizer_capability_claim: Literal[False] = False
    supports_llm_solver_claim: Literal[False] = False
    supports_self_evolution_claim: Literal[False] = False
    supports_live_model_benchmark_claim: Literal[False] = False
    compilation_ref: ArtifactRef
    batch_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=3, max_length=3)]
    family_count: Annotated[int, Field(ge=2, strict=True)]
    surface_count: Annotated[int, Field(ge=2, strict=True)]
    group_count: Annotated[int, Field(ge=1, strict=True)]
    scenario_count: Annotated[int, Field(ge=4, strict=True)]
    main_candidate_event_count: Annotated[int, Field(ge=1, strict=True)]
    repairable_event_count: Annotated[int, Field(ge=1, strict=True)]
    null_event_count: Annotated[int, Field(ge=1, strict=True)]
    unrepairable_event_count: Annotated[int, Field(ge=1, strict=True)]
    shift_hard_negative_event_count: Annotated[int, Field(ge=1, strict=True)]
    parent_pair_count: Annotated[int, Field(ge=1, strict=True)]
    revert_pair_count: Annotated[int, Field(ge=1, strict=True)]
    placebo_pair_count: Annotated[int, Field(ge=1, strict=True)]
    repair_activation_recall: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    null_overactivation_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    shift_route_accuracy: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    repairable_behavior_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    null_behavior_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    unrepairable_behavior_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    shift_hard_negative_behavior_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    adherence_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    candidate_behavior_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    oracle_behavior_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    best_constant_branch_selector_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    best_constant_output_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    best_single_family_patch_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    oracle_minus_best_constant_branch: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    oracle_minus_best_single_family_patch: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    mean_paired_gain_vs_parent: Annotated[float, Field(ge=-1.0, le=1.0, strict=True)]
    mean_paired_gain_vs_revert: Annotated[float, Field(ge=-1.0, le=1.0, strict=True)]
    mean_paired_gain_vs_placebo: Annotated[float, Field(ge=-1.0, le=1.0, strict=True)]

    @model_validator(mode="after")
    def exact_ref_media(self):
        if self.compilation_ref.media_type != HARNESS_FAULT_COMPILATION_MEDIA_TYPE:
            raise ValueError("compilation_ref declares the wrong media type")
        if any(ref.media_type != HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE for ref in self.batch_refs):
            raise ValueError("batch_refs contains the wrong media type")
        return self

    @property
    def activation_passed(self) -> bool:
        return (
            self.repair_activation_recall == 1.0
            and self.null_overactivation_rate == 0.0
            and self.shift_route_accuracy == 1.0
            and self.unrepairable_behavior_rate == 0.0
        )

    @property
    def adherence_passed(self) -> bool:
        return self.adherence_rate == 1.0

    @property
    def behavior_passed(self) -> bool:
        return (
            self.candidate_behavior_rate == self.oracle_behavior_rate
            and self.mean_paired_gain_vs_parent > 0.0
        )

    @property
    def anti_cheat_passed(self) -> bool:
        return (
            self.oracle_minus_best_constant_branch >= 0.30
            and self.oracle_minus_best_single_family_patch >= 0.30
            and self.best_constant_output_rate <= 0.05
            and self.candidate_behavior_rate > self.best_constant_branch_selector_rate
        )

    @property
    def revert_control_passed(self) -> bool:
        return self.mean_paired_gain_vs_revert > 0.0

    @property
    def placebo_control_passed(self) -> bool:
        return self.mean_paired_gain_vs_placebo > 0.0

    @property
    def mechanism_passed(self) -> bool:
        return all(
            (
                self.activation_passed,
                self.adherence_passed,
                self.behavior_passed,
                self.anti_cheat_passed,
                self.revert_control_passed,
                self.placebo_control_passed,
            )
        )


class HarnessFaultMechanismRecord(ImmutableModel):
    verification: HarnessFaultMechanismVerification
    verification_ref: ArtifactRef

    @model_validator(mode="after")
    def exact_media(self):
        if self.verification_ref.media_type != HARNESS_FAULT_MECHANISM_MEDIA_TYPE:
            raise ValueError("verification_ref declares the wrong media type")
        return self


@dataclass(frozen=True, slots=True)
class _DerivedEvent:
    scenario_id: str
    group_id: str
    scenario_role: ScenarioRole
    family: FaultFamily
    surface: FaultSurface
    oracle_branch: RuntimeBranch | None
    schedule_ref: ArtifactRef
    schedule_fingerprint: str
    comparison_kind: HarnessFaultComparisonKind
    cell: EvaluationCellKey
    seed: int
    pairing_fingerprint: str
    role: HarnessRole
    rule_id: RepairRuleId
    branch: RuntimeBranch
    adhered: bool
    behavior_correct: bool
    outcome_ref: ArtifactRef
    receipt_ref: ArtifactRef


def _load(store: ArtifactStore, ref: ArtifactRef, media: str, model: type, label: str):
    try:
        checked = ArtifactRef.model_validate(ref, strict=True)
        if checked.media_type != media:
            raise ValueError("wrong media type")
        loaded = store.get_json(checked, model)
        return model.model_validate(loaded, strict=True)
    except Exception as exc:
        raise HarnessFaultMechanismError(f"{label} artifact cannot be verified") from exc


def _derive_event(
    store: ArtifactStore,
    evaluator: HarnessFaultExplorationGrader,
    *,
    scenario: HiddenScenarioSpec,
    batch: HarnessFaultGradedBatch,
    closure: HarnessFaultScheduleClosure,
    compilation: HarnessFaultCompilationManifest,
    outcome_ref: ArtifactRef,
) -> _DerivedEvent:
    outcome = _load(
        store,
        outcome_ref,
        HARNESS_FAULT_GRADED_OUTCOME_MEDIA_TYPE,
        HarnessFaultGradedOutcome,
        "graded outcome",
    )
    receipt = _load(
        store,
        outcome.receipt_ref,
        EXECUTION_RECEIPT_MEDIA_TYPE,
        ExecutionReceipt,
        "execution receipt",
    )
    execution = _load(
        store,
        outcome.execution_ref,
        MODEL_EXECUTION_MEDIA_TYPE,
        ModelExecution,
        "model execution",
    )
    attempt = _load(
        store,
        outcome.attempt_outcome_ref,
        ATTEMPT_OUTCOME_MEDIA_TYPE,
        AttemptOutcome,
        "attempt outcome",
    )
    event = _load(
        store,
        outcome.runtime_event_ref,
        RUNTIME_EVENT_MEDIA_TYPE,
        AttestedRuntimeBranchEvent,
        "runtime event",
    )
    event = evaluator.runtime_verification.verify(event)
    preflight = _load(
        store,
        receipt.preflight_ref,
        SCHEDULE_PREFLIGHT_MEDIA_TYPE,
        SchedulePreflightCertificate,
        "preflight",
    )
    if (
        outcome.cell != receipt.cell
        or outcome.partition is not batch.partition
        or outcome.scenario_id != scenario.scenario_id
        or outcome.scenario_commitment != scenario.scenario_commitment
        or outcome.family is not scenario.family
        or outcome.surface is not scenario.surface
        or outcome.group_id != scenario.group_id
        or outcome.source_id != scenario.source_id
        or outcome.template_id != scenario.template_id
        or outcome.scenario_role is not scenario.role
    ):
        raise HarnessFaultMechanismError("outcome cluster/cell differs from hidden opening")
    if (
        receipt.schedule_fingerprint != closure.schedule.fingerprint
        or not closure.schedule.contains(receipt.cell)
        or receipt.attempt_index != 0
        or receipt.execution_ref != outcome.execution_ref
        or receipt.outcome_ref != outcome.attempt_outcome_ref
        or attempt.execution_ref != outcome.execution_ref
        or not preflight.binds_schedule(closure.schedule)
    ):
        raise HarnessFaultMechanismError("receipt/preflight does not close into schedule")
    if (
        execution.status is not ExecutionStatus.COMPLETED
        or attempt.disposition is not AttemptDisposition.SETTLED
        or execution.task != CandidateTask.from_task_view(scenario.task)
        or execution.seed != closure.schedule.seed_for(receipt.cell)
        or execution.spec != preflight.model_spec
    ):
        raise HarnessFaultMechanismError("execution/attempt differs from paired cell")
    try:
        role = compilation.role_for_harness(execution.request.harness_ref)
    except KeyError as exc:
        raise HarnessFaultMechanismError("execution harness is outside compiler") from exc
    entry = compilation.entry(role)
    HarnessMaterializer(store, spec=execution.spec).verify_execution_request(
        entry.harness_ref, execution
    )
    if (
        outcome.harness_role is not role
        or outcome.rule_id is not entry.rule_id
        or event.compilation_ref != batch.compilation_ref
        or event.request_sha256 != execution.request_sha256
        or event.task_id != execution.task_id
        or event.harness_ref != execution.request.harness_ref
        or event.rule_id is not entry.rule_id
        or event.family is not scenario.family
        or event.surface is not scenario.surface
        or event.context is not scenario.context
        or event.raw_primary_sha256 != sha256_bytes(scenario.primary.encode("utf-8"))
        or event.raw_secondary_sha256 != sha256_bytes(scenario.secondary.encode("utf-8"))
        or execution.output is None
        or event.final_output_sha256 != sha256_bytes(execution.output.encode("utf-8"))
    ):
        raise HarnessFaultMechanismError("signed runtime event differs from raw execution")
    try:
        parsed = parse_harness_fault_output(execution.output)
    except ValueError:
        parsed = None
    if parsed != outcome.parsed_output:
        raise HarnessFaultMechanismError("outcome parsed output differs from raw execution")
    branch_answer, branch_observable = evaluate_branch(
        scenario.family,
        event.branch,
        primary=scenario.primary,
        secondary=scenario.secondary,
    )
    adhered = bool(
        parsed is not None
        and parsed.answer == branch_answer
        and parsed.observable == branch_observable
    )
    behavior = bool(
        parsed is not None
        and parsed.answer == scenario.expected_answer
        and parsed.observable == scenario.expected_observable
    )
    expected_score = float(behavior)
    if outcome.observation.score != expected_score:
        raise HarnessFaultMechanismError("graded utility differs from raw behavior replay")
    return _DerivedEvent(
        scenario_id=scenario.scenario_id,
        group_id=scenario.group_id,
        scenario_role=scenario.role,
        family=scenario.family,
        surface=scenario.surface,
        oracle_branch=scenario.oracle_branch,
        schedule_ref=batch.schedule_ref,
        schedule_fingerprint=closure.schedule.fingerprint,
        comparison_kind=batch.comparison_kind,
        cell=receipt.cell,
        seed=execution.seed,
        pairing_fingerprint=receipt.cell.pairing_fingerprint,
        role=role,
        rule_id=entry.rule_id,
        branch=event.branch,
        adhered=adhered,
        behavior_correct=behavior,
        outcome_ref=outcome_ref,
        receipt_ref=outcome.receipt_ref,
    )


def _mean(values: Iterable[float | bool]) -> float:
    checked = tuple(float(value) for value in values)
    if not checked:
        raise HarnessFaultMechanismError("cannot derive a rate from no events")
    return sum(checked) / len(checked)


def _paired_gains(
    events: tuple[_DerivedEvent, ...],
    *,
    kind: HarnessFaultComparisonKind,
    expected_control: HarnessRole,
) -> tuple[float, ...]:
    selected = tuple(event for event in events if event.comparison_kind is kind)
    grouped: dict[str, list[_DerivedEvent]] = {}
    for event in selected:
        grouped.setdefault(event.pairing_fingerprint, []).append(event)
    gains: list[float] = []
    for pair in grouped.values():
        if len(pair) != 2 or {item.cell.side for item in pair} != set(EvaluationSide):
            raise HarnessFaultMechanismError("paired schedule lacks exact parent/candidate cells")
        candidate = next(item for item in pair if item.cell.side is EvaluationSide.CANDIDATE)
        control = next(item for item in pair if item.cell.side is EvaluationSide.PARENT)
        if candidate.role is not HarnessRole.CANDIDATE or control.role is not expected_control:
            raise HarnessFaultMechanismError("paired cells use the wrong compiler roles")
        if candidate.seed != control.seed:
            raise HarnessFaultMechanismError("paired candidate/control seeds differ")
        gains.append(float(candidate.behavior_correct) - float(control.behavior_correct))
    expected_pairs = len(selected) // 2
    if len(gains) != expected_pairs or expected_pairs == 0:
        raise HarnessFaultMechanismError("paired gain coverage is incomplete")
    return tuple(gains)


def verify_harness_fault_mechanism(
    store: ArtifactStore,
    evaluator: HarnessFaultExplorationGrader,
    *,
    compilation_ref: ArtifactRef,
    batch_refs: Iterable[ArtifactRef],
    attempt_ledgers: Iterable[AttemptLedger],
) -> HarnessFaultMechanismRecord:
    """Replay three complete batches against their exact live ledger writers."""

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    if type(evaluator) is not HarnessFaultExplorationGrader:
        raise TypeError("mechanism verifier requires the exact exploration grader")
    refs = tuple(ArtifactRef.model_validate(ref, strict=True) for ref in batch_refs)
    if len(refs) != 3 or len({ref.sha256 for ref in refs}) != 3:
        raise HarnessFaultMechanismError("mechanism requires exactly three distinct batch refs")
    live_ledgers = index_exact_live_ledgers(store, attempt_ledgers)
    all_live_ledgers = dict(live_ledgers)
    expected_live_tails: dict[LedgerKey, ArtifactRef] = {}
    verified_partition = verify_partition_opening(store, evaluator.partition_grant)
    scenarios = {item.task.task_id: item for item in verified_partition.scenarios}
    expected_tasks = tuple(sorted(scenarios))
    batches: dict[HarnessFaultComparisonKind, HarnessFaultGradedBatch] = {}
    closures: dict[HarnessFaultComparisonKind, HarnessFaultScheduleClosure] = {}
    all_events: list[_DerivedEvent] = []
    compilation: HarnessFaultCompilationManifest | None = None

    for ref in refs:
        batch = _load(
            store,
            ref,
            HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE,
            HarnessFaultGradedBatch,
            "graded batch",
        )
        closure = _load(
            store,
            batch.schedule_ref,
            HARNESS_FAULT_SCHEDULE_MEDIA_TYPE,
            HarnessFaultScheduleClosure,
            "schedule closure",
        )
        if batch.comparison_kind in batches:
            raise HarnessFaultMechanismError("duplicate comparison batch kind")
        if (
            batch.evaluator_fingerprint != evaluator.fingerprint
            or closure.evaluator_fingerprint != evaluator.fingerprint
            or batch.partition is not evaluator.partition_grant.partition
            or closure.partition is not evaluator.partition_grant.partition
            or batch.opening_ref != evaluator.partition_grant.opening_ref
            or batch.roster_ref != evaluator.partition_grant.roster_ref
            or batch.compilation_ref != compilation_ref
            or closure.compilation_ref != compilation_ref
            or batch.comparison_kind is not closure.comparison_kind
            or tuple(closure.schedule.task_ids) != expected_tasks
            or len(batch.outcome_refs) != closure.schedule.cell_count
            or batch.usage.schedule_fingerprint != closure.schedule.fingerprint
        ):
            raise HarnessFaultMechanismError("batch does not close exact evaluator roster/schedule")
        ledger_key, live_tail = replay_exact_batch_usage(
            store,
            batch=batch,
            closure=closure,
            live_ledgers=live_ledgers,
        )
        expected_live_tails[ledger_key] = live_tail
        outcomes = tuple(
            _load(
                store,
                outcome_ref,
                HARNESS_FAULT_GRADED_OUTCOME_MEDIA_TYPE,
                HarnessFaultGradedOutcome,
                "graded outcome",
            )
            for outcome_ref in batch.outcome_refs
        )
        if {item.receipt_ref.sha256 for item in outcomes} != {
            item.sha256 for item in batch.usage.receipt_refs
        }:
            raise HarnessFaultMechanismError("batch outcome receipts differ from trusted usage")
        cell_keys = {(item.cell.fingerprint, item.observation.seed) for item in outcomes}
        expected_cells = {
            (
                cell.fingerprint,
                closure.schedule.seed_for(cell),
            )
            for cell in closure.schedule.iter_cells()
        }
        if cell_keys != expected_cells or len(cell_keys) != len(outcomes):
            raise HarnessFaultMechanismError("batch cherry-picks or duplicates schedule cells")
        first_execution = _load(
            store,
            outcomes[0].execution_ref,
            MODEL_EXECUTION_MEDIA_TYPE,
            ModelExecution,
            "model execution",
        )
        checked_compilation = verify_fault_compilation(
            store,
            first_execution.spec,
            compilation_ref,
            expected_runtime_producer_id=evaluator.runtime_verification.producer_id,
        )
        if compilation is not None and checked_compilation != compilation:
            raise HarnessFaultMechanismError("compiler graph changes across batches")
        compilation = checked_compilation
        for outcome_ref, outcome in zip(batch.outcome_refs, outcomes, strict=True):
            try:
                scenario = scenarios[outcome.observation.task_id]
            except KeyError as exc:
                raise HarnessFaultMechanismError("outcome task is outside exact roster") from exc
            all_events.append(
                _derive_event(
                    store,
                    evaluator,
                    scenario=scenario,
                    batch=batch,
                    closure=closure,
                    compilation=checked_compilation,
                    outcome_ref=outcome_ref,
                )
            )
        batches[batch.comparison_kind] = batch
        closures[batch.comparison_kind] = closure

    if set(batches) != set(HarnessFaultComparisonKind):
        raise HarnessFaultMechanismError("main, revert, and placebo batches are all required")
    if live_ledgers or set(expected_live_tails) != set(all_live_ledgers):
        raise HarnessFaultMechanismError("each batch must consume one exact live ledger writer")
    reverify_live_ledger_tails(all_live_ledgers, expected_live_tails)
    del closures
    events = tuple(all_events)
    main_candidate = tuple(
        item
        for item in events
        if item.comparison_kind is HarnessFaultComparisonKind.MAIN
        and item.cell.side is EvaluationSide.CANDIDATE
    )
    repairable = tuple(
        item for item in main_candidate if item.scenario_role is ScenarioRole.REPAIRABLE_TARGET
    )
    null_controls = tuple(
        item for item in main_candidate if item.scenario_role is ScenarioRole.NULL_CONTROL
    )
    unrepairable = tuple(
        item for item in main_candidate if item.scenario_role is ScenarioRole.UNREPAIRABLE_CONTROL
    )
    shift_controls = tuple(
        item
        for item in main_candidate
        if item.scenario_role is ScenarioRole.DISTRACTOR_SHIFT_HARD_NEGATIVE
    )
    parent_gains = _paired_gains(
        events,
        kind=HarnessFaultComparisonKind.MAIN,
        expected_control=HarnessRole.FAULTY_PARENT,
    )
    revert_gains = _paired_gains(
        events,
        kind=HarnessFaultComparisonKind.REVERT,
        expected_control=HarnessRole.REVERT,
    )
    placebo_gains = _paired_gains(
        events,
        kind=HarnessFaultComparisonKind.PLACEBO,
        expected_control=HarnessRole.PLACEBO,
    )
    anti_cheat = anti_cheat_summary(verified_partition.scenarios)
    verification = HarnessFaultMechanismVerification(
        compilation_ref=compilation_ref,
        batch_refs=refs,
        family_count=len({item.family for item in verified_partition.scenarios}),
        surface_count=len({item.surface for item in verified_partition.scenarios}),
        group_count=len({item.group_id for item in verified_partition.scenarios}),
        scenario_count=len(verified_partition.scenarios),
        main_candidate_event_count=len(main_candidate),
        repairable_event_count=len(repairable),
        null_event_count=len(null_controls),
        unrepairable_event_count=len(unrepairable),
        shift_hard_negative_event_count=len(shift_controls),
        parent_pair_count=len(parent_gains),
        revert_pair_count=len(revert_gains),
        placebo_pair_count=len(placebo_gains),
        repair_activation_recall=_mean(item.branch is RuntimeBranch.SAFE for item in repairable),
        null_overactivation_rate=_mean(item.branch is RuntimeBranch.SAFE for item in null_controls),
        shift_route_accuracy=_mean(
            item.oracle_branch is not None and item.branch is item.oracle_branch
            for item in shift_controls
        ),
        repairable_behavior_rate=_mean(item.behavior_correct for item in repairable),
        null_behavior_rate=_mean(item.behavior_correct for item in null_controls),
        unrepairable_behavior_rate=_mean(item.behavior_correct for item in unrepairable),
        shift_hard_negative_behavior_rate=_mean(item.behavior_correct for item in shift_controls),
        adherence_rate=_mean(item.adhered for item in main_candidate),
        candidate_behavior_rate=_mean(item.behavior_correct for item in main_candidate),
        oracle_behavior_rate=anti_cheat.oracle_behavior_rate,
        best_constant_branch_selector_rate=(anti_cheat.best_constant_branch_selector_rate),
        best_constant_output_rate=anti_cheat.best_constant_output_rate,
        best_single_family_patch_rate=anti_cheat.best_single_family_patch_rate,
        oracle_minus_best_constant_branch=anti_cheat.oracle_minus_best_constant_branch,
        oracle_minus_best_single_family_patch=(anti_cheat.oracle_minus_best_single_family_patch),
        mean_paired_gain_vs_parent=_mean(parent_gains),
        mean_paired_gain_vs_revert=_mean(revert_gains),
        mean_paired_gain_vs_placebo=_mean(placebo_gains),
    )
    reverify_live_ledger_tails(all_live_ledgers, expected_live_tails)
    verification_ref = store.put_json(verification, media_type=HARNESS_FAULT_MECHANISM_MEDIA_TYPE)
    published = _load(
        store,
        verification_ref,
        HARNESS_FAULT_MECHANISM_MEDIA_TYPE,
        HarnessFaultMechanismVerification,
        "published mechanism verification",
    )
    reverify_live_ledger_tails(all_live_ledgers, expected_live_tails)
    if published != verification:
        raise HarnessFaultMechanismError("published mechanism verification changed content")
    return HarnessFaultMechanismRecord(
        verification=verification,
        verification_ref=verification_ref,
    )


__all__ = [
    "HARNESS_FAULT_MECHANISM_MEDIA_TYPE",
    "MECHANISM_VERIFIER_VERSION",
    "HarnessFaultMechanismError",
    "HarnessFaultMechanismRecord",
    "HarnessFaultMechanismVerification",
    "verify_harness_fault_mechanism",
]
