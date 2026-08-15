"""One-shot live/replay runner for a 100-call BFCL V4 public replicate."""

from __future__ import annotations

from pathlib import Path

from spiral_harness.benchmark.bfcl_v4_public_grader import (
    grade_bfcl_v4_public_prediction,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4GradingSlotBinding,
    BfclV4HoldoutUnlock,
    BfclV4PublicGraderReceipt,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    load_bfcl_v4_public_pilot,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PublicPilotTask,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan import (
    build_bfcl_v4_public_pilot_call_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
    BfclV4PilotCallSlot,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
    BfclV4ArmCandidateFreeze,
    BfclV4CallCompletion,
    BfclV4CallMaterialization,
    BfclV4CallOutcome,
)
from spiral_harness.benchmark.bfcl_v4_public_run_journal import (
    BfclV4PublicRunJournal,
    verify_bfcl_v4_public_run_closure,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import AttemptBudget, ExecutionStatus, FrozenModelSpec
from spiral_harness.execution.native_function_execution import (
    NativeFunctionBackend,
    NativeFunctionRunner,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4CandidateParseResult,
    BfclV4CandidateResolution,
    BfclV4DiagnosisParseResult,
    BfclV4DiagnosisPrompt,
    BfclV4ProposalPrompt,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    BFCL_V4_SEED_SYSTEM_PROMPT,
    build_bfcl_v4_proposal_prompt,
    parse_bfcl_v4_candidate,
    parse_bfcl_v4_diagnosis,
    resolve_bfcl_v4_candidate,
)
from spiral_harness.experiments.bfcl_v4_public_meta_native import (
    materialize_bfcl_v4_public_meta_native_request,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_CANDIDATE_MEDIA_TYPE,
    BFCL_V4_RUNNER_DIAGNOSIS_MEDIA_TYPE,
    BFCL_V4_RUNNER_GRADER_RECEIPT_MEDIA_TYPE,
    BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE,
    BFCL_V4_RUNNER_PREDICTION_MEDIA_TYPE,
    BFCL_V4_RUNNER_PUBLIC_TASK_MEDIA_TYPE,
    BFCL_V4_RUNNER_PURE_AT_B_SAMPLE_MEDIA_TYPE,
    BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    BfclV4PublicPilotRunRecord,
    BfclV4PublicPilotRunResult,
    BfclV4RunnerHarnessArtifact,
    BfclV4RunnerHarnessKind,
)
from spiral_harness.experiments.bfcl_v4_public_runner_evidence import (
    build_runner_diagnosis_prompt,
    freeze_runner_selection_inputs,
)
from spiral_harness.experiments.bfcl_v4_public_runner_holdout import (
    finish_runner_holdout_evidence,
)
from spiral_harness.experiments.bfcl_v4_public_runner_support import (
    BfclV4PublicRunnerError,
    BfclV4RunnerCallRecord,
    completed_native_response,
    load_canonical_model,
    make_harness_artifact,
    materialize_solver_request,
    prediction_from_response,
    publish_harness,
    publish_model,
    publish_native_request,
    pure_at_b_sample_from_response,
    task_and_adapter_maps,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BfclV4JointSelectionDecision,
)
from spiral_harness.providers.openai_native_contracts import NativeFunctionCallRequest
from spiral_harness.storage.protocol import ArtifactRepository


def _grade_role(slot: BfclV4PilotCallSlot) -> str:
    if slot.arm in {BfclV4PilotArm.PURE, BfclV4PilotArm.STATIC}:
        return "baseline"
    if slot.kind is BfclV4PilotCallKind.PARENT_FIT:
        return "parent-fit"
    if slot.kind is BfclV4PilotCallKind.CANDIDATE_FIT:
        return "candidate-fit"
    if slot.kind is BfclV4PilotCallKind.GATE:
        return f"gate-{slot.harness_variant}"
    if slot.kind is BfclV4PilotCallKind.HOLDOUT:
        return "holdout"
    if slot.kind is BfclV4PilotCallKind.PURE_AT_B_SAMPLE:
        return "pure-at-b-selected"
    raise BfclV4PublicRunnerError("controller slot has no grader role")


class BfclV4PublicPilotRunner:
    """Single-use orchestrator; every provider attempt consumes one frozen slot."""

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        checkout: str | Path,
        spec: FrozenModelSpec,
        backend: NativeFunctionBackend,
        attempt_budget: AttemptBudget,
        outer_seed_u64: int = 2_026_081_501,
        attempt_ledger_id: str = "bfcl-v4-public-pilot/replicate-001",
    ) -> None:
        self.repository = repository
        self.checkout = Path(checkout)
        self.spec = FrozenModelSpec.model_validate(spec, strict=True)
        self.budget = AttemptBudget.model_validate(attempt_budget, strict=True)
        if self.budget.max_attempts != 100:
            raise ValueError("BFCL public runner requires max_attempts=100")
        if self.budget.max_total_tokens < 100 * self.budget.max_tokens_per_attempt:
            raise ValueError("token budget must fit 100 fully burned one-shot reservations")
        self.plan = build_bfcl_v4_public_pilot_call_plan(outer_seed_u64)
        loaded = load_bfcl_v4_public_pilot(self.checkout)
        if loaded.manifest != BFCL_V4_PUBLIC_PILOT_MANIFEST:
            raise BfclV4PublicRunnerError("loaded public roster differs from the frozen manifest")
        self.tasks, self.adapters = task_and_adapter_maps(loaded.tasks)
        self.task_refs = tuple(
            publish_model(
                repository,
                task,
                media_type=BFCL_V4_RUNNER_PUBLIC_TASK_MEDIA_TYPE,
            )
            for task in loaded.tasks
        )
        self.ledger = AttemptLedger(
            repository,
            ledger_id=attempt_ledger_id,
            budget=self.budget,
        )
        self.native = NativeFunctionRunner(
            spec=self.spec,
            backend=backend,
            attempt_ledger=self.ledger,
        )
        self.backend = backend
        self.journal = BfclV4PublicRunJournal(repository, plan=self.plan)
        self.records: list[BfclV4RunnerCallRecord] = []
        self.resolutions: dict[BfclV4PilotArm, BfclV4CandidateResolution] = {}
        self.candidate_harnesses: dict[
            BfclV4PilotArm, tuple[BfclV4RunnerHarnessArtifact, ArtifactRef]
        ] = {}
        self.selection_decision: BfclV4JointSelectionDecision | None = None
        self.selection_evidence_ref: ArtifactRef | None = None
        self.selection_decision_ref: ArtifactRef | None = None
        self.candidate_freeze_ref: ArtifactRef | None = None
        self.joint_selection_freeze_ref: ArtifactRef | None = None
        self.tail: ArtifactRef | None = None
        self._started = False
        self.bare = make_harness_artifact(kind=BfclV4RunnerHarnessKind.BARE, system_prompt=None)
        self.static = make_harness_artifact(
            kind=BfclV4RunnerHarnessKind.STATIC,
            system_prompt=BFCL_V4_SEED_SYSTEM_PROMPT,
        )
        self.parent = make_harness_artifact(
            kind=BfclV4RunnerHarnessKind.PARENT,
            system_prompt=BFCL_V4_SEED_SYSTEM_PROMPT,
        )
        self.bare_ref = publish_harness(repository, self.bare)
        self.static_ref = publish_harness(repository, self.static)
        self.parent_ref = publish_harness(repository, self.parent)

    def run(self) -> BfclV4PublicPilotRunRecord:
        if self._started:
            raise BfclV4PublicRunnerError("BFCL public runner is single-use")
        self._started = True
        self.tail = self.journal.open()
        for slot in self.plan.calls[:10]:
            self._execute_solver(slot)
        diagnosis = self._execute_diagnoses()
        parses = self._execute_proposals(diagnosis)
        self._freeze_candidates(parses)
        for slot in self.plan.calls[14:40]:
            self._execute_solver(slot)
        self._freeze_selections()
        for slot in self.plan.calls[40:72]:
            self._execute_solver(slot)
        for slot in self.plan.calls[72:]:
            self._execute_pure_at_b_sample(slot)
        if self.selection_decision is None:
            raise BfclV4PublicRunnerError("joint selection is absent after the frozen barrier")
        holdout_ref, metrics_ref = finish_runner_holdout_evidence(
            repository=self.repository,
            checkout=self.checkout,
            plan=self.plan,
            selection_decision=self.selection_decision,
            records=tuple(self.records),
            adapters=self.adapters,
            unlock=self.journal.holdout_unlock(),
        )
        return self._close(holdout_ref=holdout_ref, metrics_ref=metrics_ref)

    def _task(self, slot: BfclV4PilotCallSlot) -> BfclV4PublicPilotTask:
        if slot.task_id is None or slot.task_id not in self.tasks:
            raise BfclV4PublicRunnerError("solver slot is not bound to a frozen public task")
        return self.tasks[slot.task_id]

    def _solver_harness(
        self,
        slot: BfclV4PilotCallSlot,
    ) -> tuple[BfclV4RunnerHarnessArtifact, ArtifactRef, str, bool]:
        if slot.arm is BfclV4PilotArm.PURE or slot.arm is BfclV4PilotArm.PURE_AT_B:
            return self.bare, self.bare_ref, "bare", False
        if slot.arm is BfclV4PilotArm.STATIC:
            return self.static, self.static_ref, "static-frozen", False
        if slot.kind is BfclV4PilotCallKind.PARENT_FIT:
            return self.parent, self.parent_ref, "parent", False
        if slot.arm not in {BfclV4PilotArm.SCORE, BfclV4PilotArm.FULL}:
            raise BfclV4PublicRunnerError("solver slot has no harness rule")
        if slot.kind is BfclV4PilotCallKind.HOLDOUT:
            if self.selection_decision is None:
                raise BfclV4PublicRunnerError("adaptive HOLDOUT precedes joint selection")
            decision = (
                self.selection_decision.score
                if slot.arm is BfclV4PilotArm.SCORE
                else self.selection_decision.full
            )
            if decision.selected_variant == "candidate":
                artifact, ref = self.candidate_harnesses[slot.arm]
                return artifact, ref, "candidate", False
            return self.parent, self.parent_ref, "parent", True
        resolution = self.resolutions.get(slot.arm)
        if resolution is None:
            raise BfclV4PublicRunnerError("adaptive evaluation precedes candidate freeze")
        if slot.kind is BfclV4PilotCallKind.CANDIDATE_FIT or (
            slot.kind is BfclV4PilotCallKind.GATE and slot.harness_variant == "candidate"
        ):
            if resolution.candidate_admissible:
                artifact, ref = self.candidate_harnesses[slot.arm]
                return artifact, ref, "candidate", False
            return self.parent, self.parent_ref, "parent", True
        if slot.kind is BfclV4PilotCallKind.GATE:
            return self.parent, self.parent_ref, "parent", False
        raise BfclV4PublicRunnerError("adaptive solver slot has no frozen execution rule")

    def _begin_native(
        self,
        *,
        slot: BfclV4PilotCallSlot,
        request: NativeFunctionCallRequest,
        harness_ref: ArtifactRef,
        executed_variant: str,
        fallback_used: bool,
        task_fingerprint: str,
    ) -> tuple[NativeFunctionCallRequest, ArtifactRef, BfclV4CallMaterialization, object]:
        if self.tail is None:
            raise BfclV4PublicRunnerError("semantic journal is not open")
        request_ref = publish_native_request(self.repository, request)
        self.tail, materialization_ref = self.journal.materialize_next_call(
            expected_tail_ref=self.tail,
            request_ref=request_ref,
            executed_harness_ref=harness_ref,
            executed_harness_variant=executed_variant,
            fallback_used=fallback_used,
        )
        materialization = load_canonical_model(
            self.repository,
            materialization_ref,
            BfclV4CallMaterialization,
            media_type=BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
        )
        previous_tail = self.ledger.tail_ref
        execution = self.native.execute_record(
            task_fingerprint=task_fingerprint,
            slot_fingerprint=canonical_sha256(slot),
            request=request,
            expected_previous_ledger_tail_ref=previous_tail,
        )
        if execution.request_ref != request_ref:
            raise BfclV4PublicRunnerError("native runner persisted a different request")
        return request, materialization_ref, materialization, execution

    def _complete(
        self,
        *,
        slot: BfclV4PilotCallSlot,
        materialization_ref: ArtifactRef,
        materialization: BfclV4CallMaterialization,
        execution,
        model_output_ref: ArtifactRef,
        prediction_ref: ArtifactRef | None,
        grader_receipt_ref: ArtifactRef | None,
        prediction=None,
        grader_receipt=None,
        pure_at_b_sample=None,
    ) -> None:
        if self.tail is None:
            raise BfclV4PublicRunnerError("semantic journal tail is absent")
        succeeded = execution.execution.status is ExecutionStatus.COMPLETED
        self.tail, completion_ref = self.journal.complete_call(
            expected_tail_ref=self.tail,
            materialization_ref=materialization_ref,
            attempt_outcome_ref=execution.outcome_ref,
            model_output_ref=model_output_ref,
            outcome=(
                BfclV4CallOutcome.SUCCEEDED if succeeded else BfclV4CallOutcome.PROVIDER_FAILURE
            ),
            prediction_ref=prediction_ref,
            grader_receipt_ref=grader_receipt_ref,
        )
        completion = load_canonical_model(
            self.repository,
            completion_ref,
            BfclV4CallCompletion,
            media_type=BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
        )
        self.records.append(
            BfclV4RunnerCallRecord(
                slot=slot,
                materialization=materialization,
                materialization_ref=materialization_ref,
                completion=completion,
                completion_ref=completion_ref,
                execution_record=execution,
                prediction=prediction,
                grader_receipt=grader_receipt,
                pure_at_b_sample=pure_at_b_sample,
            )
        )

    def _grade(
        self,
        *,
        slot: BfclV4PilotCallSlot,
        materialization: BfclV4CallMaterialization,
        prediction,
        holdout_unlock: BfclV4HoldoutUnlock | None,
    ) -> BfclV4PublicGraderReceipt:
        binding = BfclV4GradingSlotBinding(
            plan_fingerprint=self.plan.fingerprint,
            call_slot_reference_sha256=canonical_sha256(slot),
            call_id=slot.call_id,
            arm=slot.arm.value,
            grade_role=_grade_role(slot),
            intended_harness_variant=slot.harness_variant,
            executed_harness_variant=materialization.executed_harness_variant,
            fallback_used=materialization.fallback_used,
            task_id=slot.task_id,
            prediction_sha256=prediction.fingerprint,
        )
        return grade_bfcl_v4_public_prediction(
            prediction,
            binding,
            self.checkout,
            holdout_unlock=holdout_unlock,
        )

    def _execute_solver(self, slot: BfclV4PilotCallSlot) -> None:
        task = self._task(slot)
        harness, harness_ref, variant, fallback = self._solver_harness(slot)
        request = materialize_solver_request(
            task=task,
            adapter=self.adapters[task.task_id],
            harness=harness,
            spec=self.spec,
            backend=self.backend,
            seed=slot.seed_u63,
        )
        _, materialization_ref, materialization, execution = self._begin_native(
            slot=slot,
            request=request,
            harness_ref=harness_ref,
            executed_variant=variant,
            fallback_used=fallback,
            task_fingerprint=task.fingerprint,
        )
        response = completed_native_response(execution.execution)
        prediction = prediction_from_response(task.task_id, response)
        prediction_ref = publish_model(
            self.repository,
            prediction,
            media_type=BFCL_V4_RUNNER_PREDICTION_MEDIA_TYPE,
        )
        unlock = self.journal.holdout_unlock() if slot.global_slot >= 40 else None
        receipt = self._grade(
            slot=slot,
            materialization=materialization,
            prediction=prediction,
            holdout_unlock=unlock,
        )
        receipt_ref = publish_model(
            self.repository,
            receipt,
            media_type=BFCL_V4_RUNNER_GRADER_RECEIPT_MEDIA_TYPE,
        )
        self._complete(
            slot=slot,
            materialization_ref=materialization_ref,
            materialization=materialization,
            execution=execution,
            model_output_ref=prediction_ref,
            prediction_ref=prediction_ref,
            grader_receipt_ref=receipt_ref,
            prediction=prediction,
            grader_receipt=receipt,
        )

    def _execute_meta(
        self,
        *,
        slot: BfclV4PilotCallSlot,
        prompt: BfclV4DiagnosisPrompt | BfclV4ProposalPrompt,
    ):
        prompt_ref = publish_model(
            self.repository,
            prompt,
            media_type=BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE,
        )
        request = materialize_bfcl_v4_public_meta_native_request(
            prompt=prompt,
            spec=self.spec,
            backend=self.backend,
            seed=slot.seed_u63,
        )
        _, materialization_ref, materialization, execution = self._begin_native(
            slot=slot,
            request=request,
            harness_ref=prompt_ref,
            executed_variant=slot.harness_variant,
            fallback_used=False,
            task_fingerprint=prompt.fingerprint,
        )
        response = completed_native_response(execution.execution)
        if isinstance(prompt, BfclV4DiagnosisPrompt):
            parsed = parse_bfcl_v4_diagnosis(prompt, response)
            media_type = BFCL_V4_RUNNER_DIAGNOSIS_MEDIA_TYPE
        else:
            parsed = parse_bfcl_v4_candidate(prompt, response)
            media_type = BFCL_V4_RUNNER_CANDIDATE_MEDIA_TYPE
        parsed_ref = publish_model(self.repository, parsed, media_type=media_type)
        self._complete(
            slot=slot,
            materialization_ref=materialization_ref,
            materialization=materialization,
            execution=execution,
            model_output_ref=parsed_ref,
            prediction_ref=None,
            grader_receipt_ref=None,
        )
        return parsed, parsed_ref

    def _execute_diagnoses(
        self,
    ) -> dict[BfclV4PilotArm, tuple[BfclV4DiagnosisPrompt, BfclV4DiagnosisParseResult]]:
        output = {}
        for slot in self.plan.calls[10:12]:
            prompt = build_runner_diagnosis_prompt(
                arm=slot.arm,
                parent_system_prompt=BFCL_V4_SEED_SYSTEM_PROMPT,
                records=tuple(self.records),
                tasks_by_id=self.tasks,
            )
            parsed, _ = self._execute_meta(slot=slot, prompt=prompt)
            if not isinstance(parsed, BfclV4DiagnosisParseResult):
                raise BfclV4PublicRunnerError("diagnosis parser returned another contract")
            output[slot.arm] = (prompt, parsed)
        return output

    def _execute_proposals(
        self,
        diagnosis: dict[BfclV4PilotArm, tuple[BfclV4DiagnosisPrompt, BfclV4DiagnosisParseResult]],
    ) -> dict[BfclV4PilotArm, tuple[BfclV4CandidateParseResult, ArtifactRef]]:
        output = {}
        for slot in self.plan.calls[12:14]:
            diagnosis_prompt, diagnosis_result = diagnosis[slot.arm]
            proposal = build_bfcl_v4_proposal_prompt(diagnosis_prompt, diagnosis_result)
            parsed, parsed_ref = self._execute_meta(slot=slot, prompt=proposal)
            if not isinstance(parsed, BfclV4CandidateParseResult):
                raise BfclV4PublicRunnerError("candidate parser returned another contract")
            resolution = resolve_bfcl_v4_candidate(
                diagnosis_result=diagnosis_result,
                proposal_prompt=proposal,
                candidate_parse_result=parsed,
            )
            self.resolutions[slot.arm] = resolution
            output[slot.arm] = (parsed, parsed_ref)
        return output

    def _freeze_candidates(
        self,
        parses: dict[BfclV4PilotArm, tuple[BfclV4CandidateParseResult, ArtifactRef]],
    ) -> None:
        if self.tail is None or len(self.records) != 14:
            raise BfclV4PublicRunnerError("candidate freeze requires exactly fourteen calls")
        arms = {}
        for arm, proposal_index in ((BfclV4PilotArm.SCORE, 12), (BfclV4PilotArm.FULL, 13)):
            resolution = self.resolutions[arm]
            _, parse_ref = parses[arm]
            candidate_ref = None
            if resolution.candidate_admissible:
                artifact = make_harness_artifact(
                    kind=BfclV4RunnerHarnessKind.CANDIDATE,
                    system_prompt=resolution.evaluation_system_prompt,
                    arm=arm,
                )
                candidate_ref = publish_harness(self.repository, artifact)
                self.candidate_harnesses[arm] = (artifact, candidate_ref)
            arms[arm] = BfclV4ArmCandidateFreeze(
                arm=arm,
                proposal_completion_ref=self.records[proposal_index].completion_ref,
                parent_harness_ref=self.parent_ref,
                candidate_parse_ref=parse_ref,
                candidate_harness_ref=candidate_ref,
                effective_candidate_harness_ref=candidate_ref or self.parent_ref,
                candidate_valid=resolution.candidate_admissible,
                fallback_used=not resolution.candidate_admissible,
            )
        self.tail, self.candidate_freeze_ref = self.journal.freeze_candidates(
            expected_tail_ref=self.tail,
            score=arms[BfclV4PilotArm.SCORE],
            full=arms[BfclV4PilotArm.FULL],
        )

    def _freeze_selections(self) -> None:
        if self.tail is None or self.candidate_freeze_ref is None or len(self.records) != 40:
            raise BfclV4PublicRunnerError("selection freeze requires all forty search/GATE calls")
        decision, evidence_ref, decision_ref, score, full = freeze_runner_selection_inputs(
            repository=self.repository,
            plan=self.plan,
            candidate_freeze_ref=self.candidate_freeze_ref,
            records=tuple(self.records),
            resolutions=self.resolutions,
        )
        self.selection_decision = decision
        self.selection_evidence_ref = evidence_ref
        self.selection_decision_ref = decision_ref
        self.tail, self.joint_selection_freeze_ref = self.journal.freeze_selections(
            expected_tail_ref=self.tail,
            score=score,
            full=full,
        )

    def _execute_pure_at_b_sample(self, slot: BfclV4PilotCallSlot) -> None:
        task = self._task(slot)
        request = materialize_solver_request(
            task=task,
            adapter=self.adapters[task.task_id],
            harness=self.bare,
            spec=self.spec,
            backend=self.backend,
            seed=slot.seed_u63,
        )
        _, materialization_ref, materialization, execution = self._begin_native(
            slot=slot,
            request=request,
            harness_ref=self.bare_ref,
            executed_variant="bare",
            fallback_used=False,
            task_fingerprint=task.fingerprint,
        )
        response = completed_native_response(execution.execution)
        sample = pure_at_b_sample_from_response(slot, response)
        sample_ref = publish_model(
            self.repository,
            sample,
            media_type=BFCL_V4_RUNNER_PURE_AT_B_SAMPLE_MEDIA_TYPE,
        )
        self._complete(
            slot=slot,
            materialization_ref=materialization_ref,
            materialization=materialization,
            execution=execution,
            model_output_ref=sample_ref,
            prediction_ref=None,
            grader_receipt_ref=None,
            pure_at_b_sample=sample,
        )

    def _close(
        self,
        *,
        holdout_ref: ArtifactRef,
        metrics_ref: ArtifactRef,
    ) -> BfclV4PublicPilotRunRecord:
        required = (
            self.tail,
            self.candidate_freeze_ref,
            self.joint_selection_freeze_ref,
            self.selection_evidence_ref,
            self.selection_decision_ref,
        )
        if any(item is None for item in required):
            raise BfclV4PublicRunnerError("terminal runner evidence is incomplete")
        assert self.tail is not None
        self.tail, closure_ref = self.journal.close(expected_tail_ref=self.tail)
        closure_verification = verify_bfcl_v4_public_run_closure(
            self.repository,
            closure_ref,
            plan=self.plan,
        )
        state = self.ledger.state()
        if state.attempts_used != 100 or state.completed_attempts != 100:
            raise BfclV4PublicRunnerError("attempt ledger did not consume exactly 100 slots")
        if state.tail_ref is None:
            raise BfclV4PublicRunnerError("attempt ledger terminal outcome is absent")
        succeeded = sum(item.provider_succeeded for item in self.records)
        identity_observations = tuple(
            response.provider_identity_observation
            for item in self.records
            if item.provider_succeeded
            and (response := item.execution_record.execution.response) is not None
            and response.provider_identity_observation is not None
        )
        identity_coordinates = {
            (item.response_model, item.system_fingerprint) for item in identity_observations
        }
        result = BfclV4PublicPilotRunResult(
            plan_fingerprint=self.plan.fingerprint,
            schedule_content_sha256=self.plan.schedule_content_sha256,
            outer_seed_u64=self.plan.outer_seed_u64,
            manifest_fingerprint=self.plan.manifest_fingerprint,
            public_task_refs=self.task_refs,
            model_spec=self.spec,
            native_runner_fingerprint=self.native.fingerprint,
            attempt_ledger_id=self.ledger.ledger_id,
            attempt_budget=self.budget,
            attempt_ledger_tail_ref=state.tail_ref,
            attempt_outcome_refs=tuple(item.execution_record.outcome_ref for item in self.records),
            native_execution_refs=tuple(
                item.execution_record.execution_ref for item in self.records
            ),
            journal_closure_ref=closure_ref,
            closure_verification=closure_verification,
            candidate_freeze_ref=self.candidate_freeze_ref,  # type: ignore[arg-type]
            joint_selection_freeze_ref=self.joint_selection_freeze_ref,  # type: ignore[arg-type]
            joint_selection_evidence_ref=self.selection_evidence_ref,  # type: ignore[arg-type]
            joint_selection_decision_ref=self.selection_decision_ref,  # type: ignore[arg-type]
            holdout_evidence_ref=holdout_ref,
            descriptive_metrics_ref=metrics_ref,
            provider_attempts_succeeded=succeeded,
            provider_attempts_failed=100 - succeeded,
            provider_identity_observation_count=len(identity_observations),
            provider_declared_identity_consistent=(
                bool(identity_observations) and len(identity_coordinates) == 1
            ),
        )
        result_ref = publish_model(
            self.repository,
            result,
            media_type=BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
        )
        return BfclV4PublicPilotRunRecord(result=result, result_ref=result_ref)


def run_bfcl_v4_public_pilot_replicate(
    repository: ArtifactRepository,
    *,
    checkout: str | Path,
    spec: FrozenModelSpec,
    backend: NativeFunctionBackend,
    attempt_budget: AttemptBudget,
    outer_seed_u64: int = 2_026_081_501,
    attempt_ledger_id: str = "bfcl-v4-public-pilot/replicate-001",
) -> BfclV4PublicPilotRunRecord:
    """Execute exactly one frozen public/development replicate with no retry."""
    return BfclV4PublicPilotRunner(
        repository,
        checkout=checkout,
        spec=spec,
        backend=backend,
        attempt_budget=attempt_budget,
        outer_seed_u64=outer_seed_u64,
        attempt_ledger_id=attempt_ledger_id,
    ).run()
__all__ = ["BfclV4PublicPilotRunner", "run_bfcl_v4_public_pilot_replicate"]
