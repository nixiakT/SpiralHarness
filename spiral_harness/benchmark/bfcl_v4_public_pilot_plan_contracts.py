"""Fail-closed call-DAG contracts for the public/development BFCL V4 pilot."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PILOT_EXTERNAL_SEED_COMMITMENT,
    BFCL_V4_PILOT_OUTER_SEED_U64,
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PilotSplit,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

_GATE_VARIANTS = ("parent", "candidate", "revert", "placebo")
BFCL_V4_PILOT_SCHEDULE_CONTENT_SHA256 = (
    "d92f9061e2baf224d3aea8cbb1d9ca367345ab0a453ce4b106e3e5a8e2dd783e"
)


class BfclV4PilotArm(StrEnum):
    """The five model-call-budget arms in the pilot."""

    PURE = "pure"
    STATIC = "static"
    SCORE = "score"
    FULL = "full"
    PURE_AT_B = "pure-at-b"


class BfclV4PilotCallKind(StrEnum):
    """A semantic slot in the frozen call DAG."""

    BASELINE = "baseline"
    PARENT_FIT = "parent-fit"
    DIAGNOSIS = "diagnosis"
    PROPOSAL = "proposal"
    CANDIDATE_FIT = "candidate-fit"
    GATE = "gate"
    HOLDOUT = "holdout"
    PURE_AT_B_SAMPLE = "pure-at-b-sample"


class BfclV4PilotFeedbackView(StrEnum):
    """The only intentional SCORE/FULL call-input difference."""

    NONE = "none"
    SCORE_ONLY = "five-fit-aggregate-score-only"
    CANDIDATE_SAFE_FULL = "public-fit-own-response-binary-and-coarse-failure"


class BfclV4PilotCallSlot(ImmutableModel):
    """One precommitted model call in the public-pilot DAG."""

    schema_version: Literal["1"] = "1"
    global_slot: Annotated[int, Field(ge=0, lt=100, strict=True)]
    arm: BfclV4PilotArm
    arm_slot: Annotated[int, Field(ge=0, lt=28, strict=True)]
    call_id: NonEmptyStr
    kind: BfclV4PilotCallKind
    task_id: NonEmptyStr | None = None
    harness_variant: NonEmptyStr
    feedback_view: BfclV4PilotFeedbackView = BfclV4PilotFeedbackView.NONE
    seed_u63: Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)]
    depends_on: tuple[NonEmptyStr, ...] = ()
    max_provider_attempts: Literal[1] = 1
    requires_both_candidate_artifacts: bool = False
    requires_both_selection_artifacts: bool = False
    grader_feedback_available: bool = False

    @model_validator(mode="after")
    def _close_slot(self) -> Self:
        expected_call_id = (
            f"{self.arm.value}/{self.arm_slot:02d}/{self.kind.value}/"
            f"{self.task_id or 'aggregate'}/{self.harness_variant}"
        )
        if self.call_id != expected_call_id:
            raise ValueError("call ID differs from its typed coordinates")
        if self.call_id in self.depends_on or len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("call dependencies must be unique and non-reflexive")
        if self.kind in {BfclV4PilotCallKind.DIAGNOSIS, BfclV4PilotCallKind.PROPOSAL}:
            if self.task_id is not None:
                raise ValueError("diagnosis/proposal calls are aggregate, not task-bound")
        elif self.task_id is None:
            raise ValueError("evaluation and PURE@B calls must bind a public task")
        return self


def bfcl_v4_pilot_schedule_content_sha256(
    calls: tuple[BfclV4PilotCallSlot, ...],
) -> str:
    """Hash exact typed slots under the implementation-owned schedule formula."""

    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-pilot-call-schedule/v1",
            "outer_seed_u64": BFCL_V4_PILOT_OUTER_SEED_U64,
            "calls": calls,
        }
    )


def _tasks(split: BfclV4PilotSplit) -> tuple[str, ...]:
    return tuple(
        item.task_id for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster if item.split == split
    )


def _arm_calls(
    calls: tuple[BfclV4PilotCallSlot, ...],
    arm: BfclV4PilotArm,
) -> tuple[BfclV4PilotCallSlot, ...]:
    return tuple(item for item in calls if item.arm == arm)


def _expected_adaptive_coordinates() -> tuple[tuple[BfclV4PilotCallKind, str | None, str], ...]:
    fit = _tasks(BfclV4PilotSplit.FIT)
    gate = _tasks(BfclV4PilotSplit.GATE)
    holdout = _tasks(BfclV4PilotSplit.HOLDOUT)
    return (
        *((BfclV4PilotCallKind.PARENT_FIT, task_id, "parent") for task_id in fit),
        (BfclV4PilotCallKind.DIAGNOSIS, None, "diagnostic-controller"),
        (BfclV4PilotCallKind.PROPOSAL, None, "proposal-controller"),
        *((BfclV4PilotCallKind.CANDIDATE_FIT, task_id, "candidate") for task_id in fit),
        *(
            (BfclV4PilotCallKind.GATE, task_id, variant)
            for task_id in gate
            for variant in _GATE_VARIANTS
        ),
        *(
            (BfclV4PilotCallKind.HOLDOUT, task_id, "selected-or-parent-fallback")
            for task_id in holdout
        ),
    )


def _expected_pure_at_b_coordinates() -> tuple[tuple[str, str], ...]:
    return tuple(
        (task_id, f"bare-sample-{sample_index}")
        for task_index, task_id in enumerate(_tasks(BfclV4PilotSplit.HOLDOUT))
        for sample_index in range(4 if task_index < 4 else 3)
    )


def _normalized_dependencies(item: BfclV4PilotCallSlot) -> tuple[str, ...]:
    return tuple(
        value.replace("score/", "arm/").replace("full/", "arm/") for value in item.depends_on
    )


class BfclV4PublicPilotCallPlan(ImmutableModel):
    """The exact 100-call five-arm prospective plan."""

    schema_version: Literal["1"] = "1"
    manifest_fingerprint: Sha256
    outer_seed_u64: Literal[BFCL_V4_PILOT_OUTER_SEED_U64] = BFCL_V4_PILOT_OUTER_SEED_U64
    external_seed_commitment_sha256: Literal[BFCL_V4_PILOT_EXTERNAL_SEED_COMMITMENT] = (
        BFCL_V4_PILOT_EXTERNAL_SEED_COMMITMENT
    )
    external_seed_derivation_attested: Literal[False] = False
    calls: Annotated[tuple[BfclV4PilotCallSlot, ...], Field(min_length=100, max_length=100)]
    schedule_content_sha256: Literal[BFCL_V4_PILOT_SCHEDULE_CONTENT_SHA256]
    total_model_call_ceiling: Literal[100] = 100
    max_provider_attempts_per_call: Literal[1] = 1
    adaptive_stopping: Literal[False] = False
    holdout_can_continue_search: Literal[False] = False
    invalid_candidate_slot_policy: Literal["parent-fallback-consumes-all-frozen-slots"] = (
        "parent-fallback-consumes-all-frozen-slots"
    )
    invalid_candidate_selection_policy: Literal["forced-rollback"] = "forced-rollback"
    both_candidates_frozen_before_candidate_fit: Literal[True] = True
    both_arms_complete_gate_before_selection: Literal[True] = True
    both_selections_frozen_before_holdout: Literal[True] = True
    same_model_required: Literal[True] = True
    same_per_call_budget_required: Literal[True] = True
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_plan(self) -> Self:
        if self.manifest_fingerprint != BFCL_V4_PUBLIC_PILOT_MANIFEST.fingerprint:
            raise ValueError("call plan manifest fingerprint differs from the frozen roster")
        if tuple(item.global_slot for item in self.calls) != tuple(range(100)):
            raise ValueError("global call slots must be the exact sequence 0..99")
        if len({item.call_id for item in self.calls}) != 100:
            raise ValueError("pilot call IDs must be globally unique")
        known = {item.call_id for item in self.calls}
        if any(dependency not in known for item in self.calls for dependency in item.depends_on):
            raise ValueError("pilot call dependency references an unknown call")
        global_slots = {item.call_id: item.global_slot for item in self.calls}
        if any(
            global_slots[dependency] >= item.global_slot
            for item in self.calls
            for dependency in item.depends_on
        ):
            raise ValueError("each dependency must reference an earlier global slot")

        pure = _arm_calls(self.calls, BfclV4PilotArm.PURE)
        static = _arm_calls(self.calls, BfclV4PilotArm.STATIC)
        score = _arm_calls(self.calls, BfclV4PilotArm.SCORE)
        full = _arm_calls(self.calls, BfclV4PilotArm.FULL)
        pure_at_b = _arm_calls(self.calls, BfclV4PilotArm.PURE_AT_B)
        arms = (pure, static, score, full, pure_at_b)
        expected_counts = (8, 8, 28, 28, 28)
        if tuple(map(len, arms)) != expected_counts:
            raise ValueError("pilot call counts differ from the frozen five-arm allocation")
        if any(
            tuple(item.arm_slot for item in arm) != tuple(range(count))
            for arm, count in zip(arms, expected_counts, strict=True)
        ):
            raise ValueError("within-arm call slots differ from their frozen order")

        holdout = _tasks(BfclV4PilotSplit.HOLDOUT)
        expected_pure = tuple((BfclV4PilotCallKind.HOLDOUT, task_id, "bare") for task_id in holdout)
        expected_static = tuple(
            (BfclV4PilotCallKind.HOLDOUT, task_id, "static-frozen") for task_id in holdout
        )
        expected_adaptive = _expected_adaptive_coordinates()
        projections = tuple(
            tuple((item.kind, item.task_id, item.harness_variant) for item in arm)
            for arm in (pure, static, score, full)
        )
        if projections != (expected_pure, expected_static, expected_adaptive, expected_adaptive):
            raise ValueError("PURE/STATIC/SCORE/FULL task or variant sequence changed")
        pure_at_b_projection = tuple((item.task_id, item.harness_variant) for item in pure_at_b)
        if pure_at_b_projection != _expected_pure_at_b_coordinates():
            raise ValueError("PURE@B task allocation or sample variant changed")

        expected_global_order = (
            *score[:5],
            *full[:5],
            score[5],
            full[5],
            score[6],
            full[6],
            *score[7:12],
            *full[7:12],
            *score[12:20],
            *full[12:20],
            *pure,
            *static,
            *score[20:],
            *full[20:],
            *pure_at_b,
        )
        if self.calls != expected_global_order:
            raise ValueError("global order must freeze both selections before all holdout calls")

        expected_dependencies: dict[str, tuple[str, ...]] = {}
        for arm in (score, full):
            for item in arm[:5]:
                expected_dependencies[item.call_id] = ()
            expected_dependencies[arm[5].call_id] = tuple(item.call_id for item in arm[:5])
            expected_dependencies[arm[6].call_id] = (arm[5].call_id,)
        both_proposals = (score[6].call_id, full[6].call_id)
        for arm in (score, full):
            for item in arm[7:12]:
                expected_dependencies[item.call_id] = both_proposals
            own_candidate_fit = tuple(item.call_id for item in arm[7:12])
            for item in arm[12:20]:
                expected_dependencies[item.call_id] = own_candidate_fit
        both_gates = tuple(item.call_id for arm in (score, full) for item in arm[12:20])
        for arm in (pure, static, score[20:], full[20:], pure_at_b):
            for item in arm:
                expected_dependencies[item.call_id] = both_gates
        if any(item.depends_on != expected_dependencies[item.call_id] for item in self.calls):
            raise ValueError("call dependencies differ from the frozen joint-barrier DAG")

        controller_kinds = {BfclV4PilotCallKind.DIAGNOSIS, BfclV4PilotCallKind.PROPOSAL}
        for arm, expected_view in (
            (score, BfclV4PilotFeedbackView.SCORE_ONLY),
            (full, BfclV4PilotFeedbackView.CANDIDATE_SAFE_FULL),
        ):
            if any(
                item.feedback_view
                != (
                    expected_view if item.kind in controller_kinds else BfclV4PilotFeedbackView.NONE
                )
                for item in arm
            ):
                raise ValueError("adaptive feedback view differs from its frozen arm")
        if any(
            item.feedback_view != BfclV4PilotFeedbackView.NONE
            for arm in (pure, static, pure_at_b)
            for item in arm
        ):
            raise ValueError("PURE/STATIC/PURE@B cannot receive adaptive feedback")

        holdout_ids = set(holdout)
        if any(item.task_id in holdout_ids for item in self.calls[:40]):
            raise ValueError("a public holdout call appears before joint selection")
        if any(item.task_id not in holdout_ids for item in self.calls[40:]):
            raise ValueError("the final 60 slots must be public holdout calls")
        if any(item.requires_both_selection_artifacts for item in self.calls[:40]) or any(
            not item.requires_both_selection_artifacts for item in self.calls[40:]
        ):
            raise ValueError("joint selection-artifact barrier differs from the frozen schedule")

        paired_score = tuple(
            (
                item.arm_slot,
                item.kind,
                item.task_id,
                item.harness_variant,
                item.seed_u63,
                _normalized_dependencies(item),
            )
            for item in score
        )
        paired_full = tuple(
            (
                item.arm_slot,
                item.kind,
                item.task_id,
                item.harness_variant,
                item.seed_u63,
                _normalized_dependencies(item),
            )
            for item in full
        )
        if paired_score != paired_full:
            raise ValueError("SCORE/FULL task DAG, variants, or seeds are not paired")

        if any(score[index].seed_u63 != score[index + 7].seed_u63 for index in range(5)):
            raise ValueError("parent and candidate FIT seeds are not paired")
        if any(
            len({item.seed_u63 for item in arm[start : start + 4]}) != 1
            for arm in (score, full)
            for start in (12, 16)
        ):
            raise ValueError("gate variants for one task must share a seed")
        for task_index in range(8):
            matched_holdout = (
                pure[task_index],
                static[task_index],
                score[20 + task_index],
                full[20 + task_index],
            )
            if len({item.seed_u63 for item in matched_holdout}) != 1:
                raise ValueError("PURE/STATIC/SCORE/FULL holdout seeds are not paired")

        candidate_fit = tuple(
            item for item in self.calls if item.kind == BfclV4PilotCallKind.CANDIDATE_FIT
        )
        if len(candidate_fit) != 10 or any(
            not item.requires_both_candidate_artifacts for item in candidate_fit
        ):
            raise ValueError("candidate FIT calls lack the joint candidate-artifact barrier")
        if any(
            item.requires_both_candidate_artifacts
            for item in self.calls
            if item.kind != BfclV4PilotCallKind.CANDIDATE_FIT
        ):
            raise ValueError("joint candidate-artifact barrier appears on another call kind")
        controllers = tuple(item for item in self.calls if item.kind in controller_kinds)
        if len(controllers) != 4 or any(not item.grader_feedback_available for item in controllers):
            raise ValueError("adaptive controller feedback availability differs")
        if any(
            item.grader_feedback_available
            for item in self.calls
            if item.kind not in controller_kinds
        ):
            raise ValueError("evaluation calls must not receive grader feedback")
        if len({item.seed_u63 for item in pure_at_b}) != 28:
            raise ValueError("PURE@B sample seeds must be distinct")
        if self.schedule_content_sha256 != bfcl_v4_pilot_schedule_content_sha256(self.calls):
            raise ValueError("pilot schedule content fingerprint differs from typed calls")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [
    "BFCL_V4_PILOT_SCHEDULE_CONTENT_SHA256",
    "BfclV4PilotArm",
    "BfclV4PilotCallKind",
    "BfclV4PilotCallSlot",
    "BfclV4PilotFeedbackView",
    "BfclV4PublicPilotCallPlan",
    "bfcl_v4_pilot_schedule_content_sha256",
]
