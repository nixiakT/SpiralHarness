"""Fail-closed selection and descriptive-result contracts for the BFCL V4 pilot.

Only public/development evidence is representable here.  Selection observations
retain predictions, binary grades, and content-addressed execution references,
but deliberately omit possible answers and checker diagnostics.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4PublicPrediction,
    PilotTaskId,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PilotSplit,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
    BfclV4PilotCallSlot,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
    BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
    BfclV4CallCompletion,
    BfclV4CallMaterialization,
    BfclV4CallOutcome,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, Sha256
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4AdaptiveArm,
    BfclV4CandidateResolution,
)

BFCL_V4_PUBLIC_SELECTION_SCOPE = "public-development-partial-bfcl-pilot"
BFCL_V4_SELECTION_RULE = (
    "admissible-and-all-fit-gate-attempts-succeeded-and-shadow-controls-exact-"
    "and-candidate-fit-nondecreasing-and-candidate-gate-strictly-better"
)
BFCL_V4_DESCRIPTIVE_THRESHOLD_BASIS_POINTS = 1_000


def _task_ids(split: BfclV4PilotSplit) -> tuple[str, ...]:
    return tuple(
        item.task_id for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster if item.split is split
    )


BFCL_V4_FIT_TASK_IDS = _task_ids(BfclV4PilotSplit.FIT)
BFCL_V4_GATE_TASK_IDS = _task_ids(BfclV4PilotSplit.GATE)
BFCL_V4_HOLDOUT_TASK_IDS = _task_ids(BfclV4PilotSplit.HOLDOUT)
BFCL_V4_METRIC_ARM_ORDER = tuple(
    BfclV4PilotArm(value) for value in ("pure", "static", "score", "full", "pure-at-b")
)
BFCL_V4_PAIRED_CONTRASTS = tuple(
    (BfclV4PilotArm(treatment), BfclV4PilotArm(reference))
    for treatment, reference in (
        ("static", "pure"),
        ("score", "pure"),
        ("full", "pure"),
        ("pure-at-b", "pure"),
        ("score", "static"),
        ("full", "static"),
        ("full", "score"),
        ("full", "pure-at-b"),
    )
)

_ADAPTIVE_ARMS = (BfclV4PilotArm.SCORE, BfclV4PilotArm.FULL)
_GATE_VARIANTS = ("parent", "candidate", "revert", "placebo")


class _PublicDevelopmentContract(ImmutableModel):
    evidence_scope: Literal["public-development-partial-bfcl-pilot"] = (
        BFCL_V4_PUBLIC_SELECTION_SCOPE
    )
    candidate_visible: Literal[False] = False
    possible_answer_present: Literal[False] = False
    checker_diagnostics_present: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False


def _exact_ref(value: ImmutableModel, ref: ArtifactRef, media_type: str | None = None) -> bool:
    payload = canonical_json_bytes(value)
    return (
        (media_type is None or ref.media_type == media_type)
        and ref.size == len(payload)
        and ref.sha256 == sha256_bytes(payload)
    )


def _expected_selection_coordinates() -> tuple[tuple[object, str, str], ...]:
    return (
        *((BfclV4PilotCallKind.PARENT_FIT, task_id, "parent") for task_id in BFCL_V4_FIT_TASK_IDS),
        *(
            (BfclV4PilotCallKind.CANDIDATE_FIT, task_id, "candidate")
            for task_id in BFCL_V4_FIT_TASK_IDS
        ),
        *(
            (BfclV4PilotCallKind.GATE, task_id, variant)
            for task_id in BFCL_V4_GATE_TASK_IDS
            for variant in _GATE_VARIANTS
        ),
    )


class BfclV4SelectionObservation(_PublicDevelopmentContract):
    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    arm: BfclV4PilotArm
    slot: BfclV4PilotCallSlot
    slot_reference_sha256: Sha256
    materialization: BfclV4CallMaterialization
    completion: BfclV4CallCompletion
    completion_ref: ArtifactRef
    provider_attempt_succeeded: bool
    prediction_imputed_empty_for_failed_attempt: bool
    prediction: BfclV4PublicPrediction
    grader_receipt_reference_sha256: Sha256
    accepted: bool

    @model_validator(mode="after")
    def _bind_execution_and_projection(self) -> Self:
        if self.arm not in _ADAPTIVE_ARMS or self.slot.arm is not self.arm:
            raise ValueError("selection observation must belong to SCORE or FULL")
        if self.slot.kind not in {
            BfclV4PilotCallKind.PARENT_FIT,
            BfclV4PilotCallKind.CANDIDATE_FIT,
            BfclV4PilotCallKind.GATE,
        }:
            raise ValueError("selection observation is not a FIT/GATE slot")
        if self.slot_reference_sha256 != canonical_sha256(self.slot):
            raise ValueError("selection observation slot fingerprint changed")
        if self.materialization.slot != self.slot:
            raise ValueError("selection materialization belongs to another slot")
        if self.materialization.plan_fingerprint != self.plan_fingerprint:
            raise ValueError("selection materialization belongs to another plan")
        if self.materialization.call_slot_reference_sha256 != self.slot_reference_sha256:
            raise ValueError("selection materialization slot reference changed")
        if not _exact_ref(
            self.materialization,
            self.completion.materialization_ref,
            BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
        ):
            raise ValueError("completion does not bind the embedded materialization")
        if not _exact_ref(
            self.completion,
            self.completion_ref,
            BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
        ):
            raise ValueError("completion reference does not bind the embedded completion")
        completion_coordinates = (
            self.completion.plan_fingerprint,
            self.completion.call_id,
            self.completion.global_slot,
            self.completion.call_slot_reference_sha256,
        )
        slot_coordinates = (
            self.plan_fingerprint,
            self.slot.call_id,
            self.slot.global_slot,
            self.slot_reference_sha256,
        )
        if completion_coordinates != slot_coordinates:
            raise ValueError("selection completion coordinates differ from the slot")

        succeeded = self.completion.outcome is BfclV4CallOutcome.SUCCEEDED
        if self.provider_attempt_succeeded is not succeeded:
            raise ValueError("provider-success projection differs from the completion")
        if self.completion.prediction_ref is None or self.completion.grader_receipt_ref is None:
            raise ValueError("gradable selection completion requires prediction and grade")
        if not _exact_ref(self.prediction, self.completion.prediction_ref):
            raise ValueError("selection prediction reference changed")
        if self.prediction.task_id != self.slot.task_id:
            raise ValueError("selection prediction belongs to another task")
        if self.grader_receipt_reference_sha256 != self.completion.grader_receipt_ref.sha256:
            raise ValueError("selection grade projection belongs to another receipt")
        if self.prediction_imputed_empty_for_failed_attempt is not (not succeeded):
            raise ValueError("empty-prediction imputation flag differs from provider outcome")
        if not succeeded and (self.prediction.calls or self.accepted):
            raise ValueError("provider failure requires an imputed empty incorrect prediction")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4ArmSelectionEvidence(_PublicDevelopmentContract):
    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    arm: BfclV4PilotArm
    candidate_freeze_ref: ArtifactRef
    candidate_resolution: BfclV4CandidateResolution
    candidate_resolution_fingerprint: Sha256
    observations: Annotated[
        tuple[BfclV4SelectionObservation, ...], Field(min_length=18, max_length=18)
    ]
    exact_frozen_arm_slots_present: Literal[True] = True

    @model_validator(mode="after")
    def _close_arm_evidence(self) -> Self:
        if self.arm not in _ADAPTIVE_ARMS:
            raise ValueError("selection evidence arm must be SCORE or FULL")
        expected_adaptive = {
            BfclV4PilotArm.SCORE: BfclV4AdaptiveArm.SCORE,
            BfclV4PilotArm.FULL: BfclV4AdaptiveArm.FULL,
        }[self.arm]
        if self.candidate_resolution.arm is not expected_adaptive:
            raise ValueError("candidate resolution belongs to another arm")
        if self.candidate_resolution_fingerprint != self.candidate_resolution.fingerprint:
            raise ValueError("candidate resolution fingerprint changed")
        if self.candidate_freeze_ref.media_type != BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE:
            raise ValueError("candidate freeze reference has the wrong media type")
        coordinates = tuple(
            (item.slot.kind, item.slot.task_id, item.slot.harness_variant)
            for item in self.observations
        )
        if coordinates != _expected_selection_coordinates():
            raise ValueError("arm evidence differs from the exact FIT/GATE slot roster")
        if len({item.completion_ref.sha256 for item in self.observations}) != 18:
            raise ValueError("arm selection completions must not repeat")

        for item in self.observations:
            if (
                item.plan_fingerprint != self.plan_fingerprint
                or item.schedule_content_sha256 != self.schedule_content_sha256
                or item.arm is not self.arm
            ):
                raise ValueError("arm observation differs from its plan or arm evidence")
            materialization = item.materialization
            if item.slot.kind is BfclV4PilotCallKind.PARENT_FIT:
                expected_execution = ("parent", False, None)
            elif item.slot.kind is BfclV4PilotCallKind.CANDIDATE_FIT or (
                item.slot.kind is BfclV4PilotCallKind.GATE
                and item.slot.harness_variant == "candidate"
            ):
                expected_execution = (
                    self.candidate_resolution.executed_harness_variant,
                    self.candidate_resolution.exact_parent_fallback_used,
                    self.candidate_freeze_ref,
                )
            else:
                expected_execution = ("parent", False, self.candidate_freeze_ref)
            actual_execution = (
                materialization.executed_harness_variant,
                materialization.fallback_used,
                materialization.candidate_freeze_ref,
            )
            if actual_execution != expected_execution:
                raise ValueError("selection observation violates candidate fallback policy")
            if materialization.joint_selection_freeze_ref is not None:
                raise ValueError("FIT/GATE evidence cannot depend on a later selection")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4JointSelectionEvidence(_PublicDevelopmentContract):
    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    candidate_freeze_ref: ArtifactRef
    score: BfclV4ArmSelectionEvidence
    full: BfclV4ArmSelectionEvidence
    global_gate_completion_refs: Annotated[
        tuple[ArtifactRef, ...], Field(min_length=16, max_length=16)
    ]
    all_sixteen_gate_completions_present: Literal[True] = True
    both_arm_evidence_sets_complete: Literal[True] = True
    runner_journal_completion_barrier_required: Literal[True] = True
    barrier_ordering_independently_attested: Literal[False] = False

    @model_validator(mode="after")
    def _close_joint_evidence(self) -> Self:
        if self.score.arm is not BfclV4PilotArm.SCORE or self.full.arm is not BfclV4PilotArm.FULL:
            raise ValueError("joint selection evidence must contain SCORE then FULL")
        for arm in (self.score, self.full):
            if (
                arm.plan_fingerprint != self.plan_fingerprint
                or arm.schedule_content_sha256 != self.schedule_content_sha256
                or arm.candidate_freeze_ref != self.candidate_freeze_ref
            ):
                raise ValueError("joint selection arm evidence differs from its lineage")
        expected_gate_refs = tuple(
            item.completion_ref
            for arm in (self.score, self.full)
            for item in arm.observations
            if item.slot.kind is BfclV4PilotCallKind.GATE
        )
        if self.global_gate_completion_refs != expected_gate_refs:
            raise ValueError("joint evidence does not bind the exact sixteen GATE completions")
        if len({item.sha256 for item in self.global_gate_completion_refs}) != 16:
            raise ValueError("global GATE completion references must not repeat")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4SelectionDecision(StrEnum):
    PROMOTE = "promote"
    ROLLBACK = "rollback"


class BfclV4RollbackReason(StrEnum):
    CANDIDATE_INVALID = "candidate-invalid"
    PROVIDER_ATTEMPT_FAILURE = "provider-attempt-failure"
    SHADOW_CONTROL_MISMATCH = "shadow-control-mismatch"
    CANDIDATE_FIT_REGRESSION = "candidate-fit-regression"
    CANDIDATE_GATE_NOT_STRICTLY_BETTER = "candidate-gate-not-strictly-better"


class BfclV4ArmSelectionDecision(_PublicDevelopmentContract):
    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    arm: BfclV4PilotArm
    candidate_freeze_reference_sha256: Sha256
    joint_evidence_fingerprint: Sha256
    arm_evidence_fingerprint: Sha256
    candidate_resolution_fingerprint: Sha256
    candidate_admissible: bool
    all_fit_gate_provider_attempts_succeeded: bool
    shadow_controls_exact: bool
    parent_fit_correct: Annotated[int, Field(ge=0, le=5, strict=True)]
    candidate_fit_correct: Annotated[int, Field(ge=0, le=5, strict=True)]
    parent_gate_correct: Annotated[int, Field(ge=0, le=2, strict=True)]
    candidate_gate_correct: Annotated[int, Field(ge=0, le=2, strict=True)]
    candidate_fit_nondecreasing: bool
    candidate_gate_strictly_better: bool
    decision: BfclV4SelectionDecision
    selected_variant: Literal["parent", "candidate"]
    parent_system_prompt_sha256: Sha256
    candidate_system_prompt_sha256: Sha256 | None
    selected_system_prompt_sha256: Sha256
    forced_rollback: bool
    rollback_reasons: tuple[BfclV4RollbackReason, ...]
    selection_rule: Literal[
        "admissible-and-all-fit-gate-attempts-succeeded-and-shadow-controls-exact-"
        "and-candidate-fit-nondecreasing-and-candidate-gate-strictly-better"
    ] = BFCL_V4_SELECTION_RULE
    manual_override: Literal[False] = False

    @model_validator(mode="after")
    def _close_selection_rule(self) -> Self:
        if self.arm not in _ADAPTIVE_ARMS:
            raise ValueError("selection decision arm must be SCORE or FULL")
        if self.candidate_fit_nondecreasing != (
            self.candidate_fit_correct >= self.parent_fit_correct
        ):
            raise ValueError("FIT nondecreasing flag differs from exact counts")
        if self.candidate_gate_strictly_better != (
            self.candidate_gate_correct > self.parent_gate_correct
        ):
            raise ValueError("GATE improvement flag differs from exact counts")
        expected_reasons = tuple(
            reason
            for condition, reason in (
                (not self.candidate_admissible, BfclV4RollbackReason.CANDIDATE_INVALID),
                (
                    not self.all_fit_gate_provider_attempts_succeeded,
                    BfclV4RollbackReason.PROVIDER_ATTEMPT_FAILURE,
                ),
                (not self.shadow_controls_exact, BfclV4RollbackReason.SHADOW_CONTROL_MISMATCH),
                (
                    not self.candidate_fit_nondecreasing,
                    BfclV4RollbackReason.CANDIDATE_FIT_REGRESSION,
                ),
                (
                    not self.candidate_gate_strictly_better,
                    BfclV4RollbackReason.CANDIDATE_GATE_NOT_STRICTLY_BETTER,
                ),
            )
            if condition
        )
        if self.rollback_reasons != expected_reasons:
            raise ValueError("rollback reasons differ from the frozen rule")
        promote = not expected_reasons
        expected_decision = (
            BfclV4SelectionDecision.PROMOTE if promote else BfclV4SelectionDecision.ROLLBACK
        )
        if self.decision is not expected_decision:
            raise ValueError("selection decision differs from the frozen rule")
        if self.forced_rollback is not (not self.candidate_admissible):
            raise ValueError("only an invalid candidate forces rollback")
        if self.candidate_admissible != (self.candidate_system_prompt_sha256 is not None):
            raise ValueError("candidate prompt identity differs from admissibility")
        expected_variant = "candidate" if promote else "parent"
        expected_prompt = (
            self.candidate_system_prompt_sha256 if promote else self.parent_system_prompt_sha256
        )
        if self.selected_variant != expected_variant or self.selected_system_prompt_sha256 != (
            expected_prompt
        ):
            raise ValueError("selected prompt does not follow promote-or-rollback")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4JointSelectionDecision(_PublicDevelopmentContract):
    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    candidate_freeze_reference_sha256: Sha256
    joint_evidence_fingerprint: Sha256
    global_gate_completion_refs: Annotated[
        tuple[ArtifactRef, ...], Field(min_length=16, max_length=16)
    ]
    score: BfclV4ArmSelectionDecision
    full: BfclV4ArmSelectionDecision
    all_sixteen_gate_completions_bound: Literal[True] = True
    both_decisions_computed_from_joint_evidence: Literal[True] = True
    runner_journal_completion_barrier_required: Literal[True] = True
    barrier_ordering_independently_attested: Literal[False] = False
    holdout_can_continue_search: Literal[False] = False
    public_development_only: Literal[True] = True

    @model_validator(mode="after")
    def _close_joint_decision(self) -> Self:
        if self.score.arm is not BfclV4PilotArm.SCORE or self.full.arm is not BfclV4PilotArm.FULL:
            raise ValueError("joint decision must contain SCORE then FULL")
        for decision in (self.score, self.full):
            if (
                decision.plan_fingerprint != self.plan_fingerprint
                or decision.schedule_content_sha256 != self.schedule_content_sha256
                or decision.candidate_freeze_reference_sha256
                != self.candidate_freeze_reference_sha256
                or decision.joint_evidence_fingerprint != self.joint_evidence_fingerprint
            ):
                raise ValueError("arm decision differs from the joint decision lineage")
        if len({item.sha256 for item in self.global_gate_completion_refs}) != 16:
            raise ValueError("joint decision GATE references must not repeat")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4HoldoutObservation(_PublicDevelopmentContract):
    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    joint_selection_decision_fingerprint: Sha256
    arm_selection_decision_fingerprint: Sha256 | None = None
    task_id: PilotTaskId
    arm: BfclV4PilotArm
    source_call_ids: Annotated[tuple[str, ...], Field(min_length=1, max_length=4)]
    source_completion_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1, max_length=4)]
    source_provider_attempt_succeeded: Annotated[
        tuple[bool, ...], Field(min_length=1, max_length=4)
    ]
    pure_at_b_selection_fingerprint: Sha256 | None = None
    prediction_imputed_empty_for_failed_attempt: bool
    selected_prediction: BfclV4PublicPrediction
    selected_prediction_ref: ArtifactRef
    grader_receipt_ref: ArtifactRef
    holdout_unlock_fingerprint: Sha256
    accepted: bool

    @model_validator(mode="after")
    def _close_holdout_observation(self) -> Self:
        if self.task_id not in BFCL_V4_HOLDOUT_TASK_IDS or self.arm not in BFCL_V4_METRIC_ARM_ORDER:
            raise ValueError("holdout observation is outside the frozen task-arm roster")
        expected_count = 1
        if self.arm is BfclV4PilotArm.PURE_AT_B:
            task_index = BFCL_V4_HOLDOUT_TASK_IDS.index(self.task_id)
            expected_count = 4 if task_index < 4 else 3
        if not (
            len(self.source_call_ids)
            == len(self.source_completion_refs)
            == len(self.source_provider_attempt_succeeded)
            == expected_count
        ):
            raise ValueError("holdout source evidence count differs from the frozen arm budget")
        if (
            len(set(self.source_call_ids)) != expected_count
            or len({item.sha256 for item in self.source_completion_refs}) != expected_count
        ):
            raise ValueError("holdout source calls or completions must not repeat")
        if any(
            item.media_type != BFCL_V4_CALL_COMPLETION_MEDIA_TYPE
            for item in self.source_completion_refs
        ):
            raise ValueError("holdout source completion has the wrong media type")

        if self.selected_prediction.task_id != self.task_id or not _exact_ref(
            self.selected_prediction, self.selected_prediction_ref
        ):
            raise ValueError("HOLDOUT selected prediction or reference differs from its task")
        if self.arm is BfclV4PilotArm.PURE_AT_B:
            failed = tuple(not item for item in self.source_provider_attempt_succeeded)
            if self.pure_at_b_selection_fingerprint is None:
                raise ValueError("PURE@B requires target-free selection and final grade evidence")
            if all(failed) and not self.prediction_imputed_empty_for_failed_attempt:
                raise ValueError("all-failed PURE@B must grade an imputed empty prediction")
            if self.prediction_imputed_empty_for_failed_attempt and (
                not any(failed) or self.selected_prediction.calls or self.accepted
            ):
                raise ValueError("failed PURE@B selection requires an empty incorrect prediction")
        else:
            if self.pure_at_b_selection_fingerprint is not None:
                raise ValueError("non-PURE@B observation cannot bind a plurality selection")
            succeeded = self.source_provider_attempt_succeeded == (True,)
            if self.prediction_imputed_empty_for_failed_attempt is not (not succeeded):
                raise ValueError("HOLDOUT imputation flag differs from provider outcome")
            if not succeeded and (self.selected_prediction.calls or self.accepted):
                raise ValueError("failed HOLDOUT attempt requires an empty incorrect prediction")
        if self.arm in _ADAPTIVE_ARMS:
            if self.arm_selection_decision_fingerprint is None:
                raise ValueError("adaptive HOLDOUT must bind its arm selection")
        elif self.arm_selection_decision_fingerprint is not None:
            raise ValueError("nonadaptive HOLDOUT cannot bind an arm selection")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4HoldoutEvidence(_PublicDevelopmentContract):
    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    joint_selection_decision_fingerprint: Sha256
    observations: Annotated[
        tuple[BfclV4HoldoutObservation, ...], Field(min_length=40, max_length=40)
    ]
    all_sixty_holdout_call_completions_present: Literal[True] = True
    holdout_can_continue_search: Literal[False] = False
    first_inspection_makes_holdout_development_data: Literal[True] = True

    @model_validator(mode="after")
    def _close_holdout_matrix(self) -> Self:
        expected_keys = tuple(
            (arm, task_id)
            for arm in BFCL_V4_METRIC_ARM_ORDER
            for task_id in BFCL_V4_HOLDOUT_TASK_IDS
        )
        actual_keys = tuple((item.arm, item.task_id) for item in self.observations)
        if actual_keys != expected_keys:
            raise ValueError("HOLDOUT observations differ from the exact arm-major matrix")
        completion_hashes: list[str] = []
        for item in self.observations:
            if (
                item.plan_fingerprint != self.plan_fingerprint
                or item.schedule_content_sha256 != self.schedule_content_sha256
                or item.joint_selection_decision_fingerprint
                != self.joint_selection_decision_fingerprint
            ):
                raise ValueError("HOLDOUT observation differs from its result lineage")
            completion_hashes.extend(ref.sha256 for ref in item.source_completion_refs)
        if len(completion_hashes) != 60 or len(set(completion_hashes)) != 60:
            raise ValueError("HOLDOUT evidence must bind sixty distinct call completions")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4ArmDescriptiveMetric(ImmutableModel):
    arm: BfclV4PilotArm
    task_ids: tuple[PilotTaskId, ...]
    correctness: tuple[bool, ...]
    correct_count: Annotated[int, Field(ge=0, le=8, strict=True)]
    accuracy_basis_points: Annotated[int, Field(ge=0, le=10_000, multiple_of=1_250, strict=True)]

    @model_validator(mode="after")
    def _close_accuracy(self) -> Self:
        if self.arm not in BFCL_V4_METRIC_ARM_ORDER:
            raise ValueError("descriptive metric uses an unknown arm")
        if self.task_ids != BFCL_V4_HOLDOUT_TASK_IDS or len(self.correctness) != 8:
            raise ValueError("descriptive metric differs from the frozen HOLDOUT roster")
        correct = sum(self.correctness)
        if self.correct_count != correct or self.accuracy_basis_points != correct * 1_250:
            raise ValueError("accuracy basis points differ from the exact binary vector")
        return self


class BfclV4PairedDescriptiveDelta(ImmutableModel):
    treatment_arm: BfclV4PilotArm
    reference_arm: BfclV4PilotArm
    task_ids: tuple[PilotTaskId, ...]
    treatment_correctness: tuple[bool, ...]
    reference_correctness: tuple[bool, ...]
    task_count: Literal[8] = 8
    wins: Annotated[int, Field(ge=0, le=8, strict=True)]
    ties: Annotated[int, Field(ge=0, le=8, strict=True)]
    losses: Annotated[int, Field(ge=0, le=8, strict=True)]
    delta_basis_points: Annotated[int, Field(ge=-10_000, le=10_000, multiple_of=1_250, strict=True)]
    descriptive_threshold_basis_points: Literal[1_000] = BFCL_V4_DESCRIPTIVE_THRESHOLD_BASIS_POINTS
    strictly_exceeds_positive_ten_percentage_points: bool
    descriptive_flag_only: Literal[True] = True
    statistical_significance_claimed: Literal[False] = False

    @model_validator(mode="after")
    def _close_paired_delta(self) -> Self:
        if (self.treatment_arm, self.reference_arm) not in BFCL_V4_PAIRED_CONTRASTS:
            raise ValueError("paired delta is outside the frozen contrast roster")
        if (
            self.task_ids != BFCL_V4_HOLDOUT_TASK_IDS
            or len(self.treatment_correctness) != 8
            or len(self.reference_correctness) != 8
        ):
            raise ValueError("paired delta differs from the frozen HOLDOUT roster")
        wins = sum(
            treatment and not reference
            for treatment, reference in zip(
                self.treatment_correctness, self.reference_correctness, strict=True
            )
        )
        losses = sum(
            reference and not treatment
            for treatment, reference in zip(
                self.treatment_correctness, self.reference_correctness, strict=True
            )
        )
        ties = 8 - wins - losses
        delta = (wins - losses) * 1_250
        if (self.wins, self.ties, self.losses, self.delta_basis_points) != (
            wins,
            ties,
            losses,
            delta,
        ):
            raise ValueError("paired counts or basis-point delta differ from binary vectors")
        if self.strictly_exceeds_positive_ten_percentage_points is not (delta > 1_000):
            raise ValueError("descriptive >10 pp flag differs from the exact paired delta")
        return self


class BfclV4PublicDescriptiveMetrics(_PublicDevelopmentContract):
    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    joint_selection_decision_fingerprint: Sha256
    holdout_evidence_fingerprint: Sha256
    arms: Annotated[tuple[BfclV4ArmDescriptiveMetric, ...], Field(min_length=5, max_length=5)]
    paired_deltas: Annotated[
        tuple[BfclV4PairedDescriptiveDelta, ...], Field(min_length=8, max_length=8)
    ]
    integer_basis_points_only: Literal[True] = True
    descriptive_threshold_strictly_greater_than_basis_points: Literal[1_000] = 1_000
    public_development_descriptive_only: Literal[True] = True
    holdout_now_development_data: Literal[True] = True
    multiplicity_adjusted_inference_available: Literal[False] = False
    confidence_interval_available: Literal[False] = False
    statistical_significance_claimed: Literal[False] = False
    sealed_evidence: Literal[False] = False
    official_full_suite: Literal[False] = False

    @model_validator(mode="after")
    def _close_descriptive_summary(self) -> Self:
        if tuple(item.arm for item in self.arms) != BFCL_V4_METRIC_ARM_ORDER:
            raise ValueError("descriptive arm metrics differ from the frozen order")
        if (
            tuple((item.treatment_arm, item.reference_arm) for item in self.paired_deltas)
            != BFCL_V4_PAIRED_CONTRASTS
        ):
            raise ValueError("paired descriptive deltas differ from the frozen order")
        by_arm = {item.arm: item for item in self.arms}
        for delta in self.paired_deltas:
            if (
                delta.treatment_correctness != by_arm[delta.treatment_arm].correctness
                or delta.reference_correctness != by_arm[delta.reference_arm].correctness
            ):
                raise ValueError("paired delta vectors differ from their arm metrics")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
