"""Deterministic 100-call schedule for the public/development BFCL V4 pilot."""

from __future__ import annotations

import hashlib

from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PILOT_OUTER_SEED_U64,
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PilotSplit,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BFCL_V4_PILOT_OUTER_SEEDS_U64,
    BfclV4PilotArm,
    BfclV4PilotCallKind,
    BfclV4PilotCallSlot,
    BfclV4PilotFeedbackView,
    BfclV4PilotOuterSeed,
    BfclV4PublicPilotCallPlan,
    bfcl_v4_pilot_schedule_content_sha256,
)
from spiral_harness.core.canonical import canonical_json_bytes

_GATE_VARIANTS = ("parent", "candidate", "revert", "placebo")
_SEED_MASK = (1 << 63) - 1


def _checked_outer_seed(outer_seed_u64: int) -> BfclV4PilotOuterSeed:
    if type(outer_seed_u64) is not int or outer_seed_u64 not in BFCL_V4_PILOT_OUTER_SEEDS_U64:
        raise ValueError("outer seed is absent from the frozen three-replicate campaign")
    return outer_seed_u64


def _seed_u63(outer_seed_u64: BfclV4PilotOuterSeed, *coordinate: object) -> int:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "spiral-bfcl-v4-public-pilot-model-seed/v1",
                "outer_seed_u64": outer_seed_u64,
                "coordinate": coordinate,
            }
        )
    ).digest()
    # Match derive_seed_v2: providers commonly require a signed int64 seed.
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


def _call_id(
    arm: BfclV4PilotArm,
    arm_slot: int,
    kind: BfclV4PilotCallKind,
    task_id: str | None,
    variant: str,
) -> str:
    return f"{arm.value}/{arm_slot:02d}/{kind.value}/{task_id or 'aggregate'}/{variant}"


def _slot(
    *,
    outer_seed_u64: BfclV4PilotOuterSeed,
    arm: BfclV4PilotArm,
    arm_slot: int,
    kind: BfclV4PilotCallKind,
    task_id: str | None,
    variant: str,
    seed_coordinate: tuple[object, ...],
    depends_on: tuple[str, ...] = (),
    feedback_view: BfclV4PilotFeedbackView = BfclV4PilotFeedbackView.NONE,
    requires_both_candidate_artifacts: bool = False,
    requires_both_selection_artifacts: bool = False,
    grader_feedback_available: bool = False,
) -> BfclV4PilotCallSlot:
    return BfclV4PilotCallSlot(
        global_slot=0,
        arm=arm,
        arm_slot=arm_slot,
        call_id=_call_id(arm, arm_slot, kind, task_id, variant),
        kind=kind,
        task_id=task_id,
        harness_variant=variant,
        feedback_view=feedback_view,
        seed_u63=_seed_u63(outer_seed_u64, *seed_coordinate),
        depends_on=depends_on,
        requires_both_candidate_artifacts=requires_both_candidate_artifacts,
        requires_both_selection_artifacts=requires_both_selection_artifacts,
        grader_feedback_available=grader_feedback_available,
    )


def _tasks(split: BfclV4PilotSplit) -> tuple[object, ...]:
    return tuple(item for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster if item.split == split)


def _all_gate_ids() -> tuple[str, ...]:
    gate = _tasks(BfclV4PilotSplit.GATE)
    return tuple(
        _call_id(arm, slot, BfclV4PilotCallKind.GATE, item.task_id, variant)
        for arm in (BfclV4PilotArm.SCORE, BfclV4PilotArm.FULL)
        for slot, (item, variant) in enumerate(
            ((item, variant) for item in gate for variant in _GATE_VARIANTS),
            start=12,
        )
    )


def _baseline_slots(
    arm: BfclV4PilotArm,
    variant: str,
    outer_seed_u64: BfclV4PilotOuterSeed,
) -> tuple[BfclV4PilotCallSlot, ...]:
    return tuple(
        _slot(
            outer_seed_u64=outer_seed_u64,
            arm=arm,
            arm_slot=index,
            kind=BfclV4PilotCallKind.HOLDOUT,
            task_id=item.task_id,
            variant=variant,
            seed_coordinate=("matched-holdout", item.task_id),
            depends_on=_all_gate_ids(),
            requires_both_selection_artifacts=True,
        )
        for index, item in enumerate(_tasks(BfclV4PilotSplit.HOLDOUT))
    )


def _adaptive_slots(
    arm: BfclV4PilotArm,
    outer_seed_u64: BfclV4PilotOuterSeed,
) -> tuple[BfclV4PilotCallSlot, ...]:
    fit = _tasks(BfclV4PilotSplit.FIT)
    gate = _tasks(BfclV4PilotSplit.GATE)
    holdout = _tasks(BfclV4PilotSplit.HOLDOUT)
    view = (
        BfclV4PilotFeedbackView.SCORE_ONLY
        if arm == BfclV4PilotArm.SCORE
        else BfclV4PilotFeedbackView.CANDIDATE_SAFE_FULL
    )
    calls: list[BfclV4PilotCallSlot] = []
    for index, item in enumerate(fit):
        calls.append(
            _slot(
                outer_seed_u64=outer_seed_u64,
                arm=arm,
                arm_slot=index,
                kind=BfclV4PilotCallKind.PARENT_FIT,
                task_id=item.task_id,
                variant="parent",
                seed_coordinate=("adaptive-fit", item.task_id),
            )
        )
    parent_ids = tuple(item.call_id for item in calls)
    calls.append(
        _slot(
            outer_seed_u64=outer_seed_u64,
            arm=arm,
            arm_slot=5,
            kind=BfclV4PilotCallKind.DIAGNOSIS,
            task_id=None,
            variant="diagnostic-controller",
            seed_coordinate=("adaptive-controller", "diagnosis"),
            depends_on=parent_ids,
            feedback_view=view,
            grader_feedback_available=True,
        )
    )
    calls.append(
        _slot(
            outer_seed_u64=outer_seed_u64,
            arm=arm,
            arm_slot=6,
            kind=BfclV4PilotCallKind.PROPOSAL,
            task_id=None,
            variant="proposal-controller",
            seed_coordinate=("adaptive-controller", "proposal"),
            depends_on=(calls[-1].call_id,),
            feedback_view=view,
            grader_feedback_available=True,
        )
    )
    both_proposals = tuple(
        _call_id(candidate_arm, 6, BfclV4PilotCallKind.PROPOSAL, None, "proposal-controller")
        for candidate_arm in (BfclV4PilotArm.SCORE, BfclV4PilotArm.FULL)
    )
    for index, item in enumerate(fit, start=7):
        calls.append(
            _slot(
                outer_seed_u64=outer_seed_u64,
                arm=arm,
                arm_slot=index,
                kind=BfclV4PilotCallKind.CANDIDATE_FIT,
                task_id=item.task_id,
                variant="candidate",
                seed_coordinate=("adaptive-fit", item.task_id),
                depends_on=both_proposals,
                requires_both_candidate_artifacts=True,
            )
        )
    candidate_ids = tuple(
        item.call_id for item in calls if item.kind == BfclV4PilotCallKind.CANDIDATE_FIT
    )
    gate_slot = 12
    for item in gate:
        for variant in _GATE_VARIANTS:
            calls.append(
                _slot(
                    outer_seed_u64=outer_seed_u64,
                    arm=arm,
                    arm_slot=gate_slot,
                    kind=BfclV4PilotCallKind.GATE,
                    task_id=item.task_id,
                    variant=variant,
                    seed_coordinate=("adaptive-gate", item.task_id),
                    depends_on=candidate_ids,
                )
            )
            gate_slot += 1
    both_gate_ids = _all_gate_ids()
    for index, item in enumerate(holdout, start=20):
        calls.append(
            _slot(
                outer_seed_u64=outer_seed_u64,
                arm=arm,
                arm_slot=index,
                kind=BfclV4PilotCallKind.HOLDOUT,
                task_id=item.task_id,
                variant="selected-or-parent-fallback",
                seed_coordinate=("matched-holdout", item.task_id),
                depends_on=both_gate_ids,
                requires_both_selection_artifacts=True,
            )
        )
    return tuple(calls)


def _pure_at_b_slots(
    outer_seed_u64: BfclV4PilotOuterSeed,
) -> tuple[BfclV4PilotCallSlot, ...]:
    calls: list[BfclV4PilotCallSlot] = []
    for task_index, item in enumerate(_tasks(BfclV4PilotSplit.HOLDOUT)):
        count = 4 if task_index < 4 else 3
        for sample_index in range(count):
            arm_slot = len(calls)
            calls.append(
                _slot(
                    outer_seed_u64=outer_seed_u64,
                    arm=BfclV4PilotArm.PURE_AT_B,
                    arm_slot=arm_slot,
                    kind=BfclV4PilotCallKind.PURE_AT_B_SAMPLE,
                    task_id=item.task_id,
                    variant=f"bare-sample-{sample_index}",
                    seed_coordinate=("pure-at-b", item.task_id, sample_index),
                    depends_on=_all_gate_ids(),
                    requires_both_selection_artifacts=True,
                )
            )
    return tuple(calls)


def build_bfcl_v4_public_pilot_call_plan(
    outer_seed_u64: int = BFCL_V4_PILOT_OUTER_SEED_U64,
) -> BfclV4PublicPilotCallPlan:
    """Build one frozen 100-call replicate and its dependency barriers."""

    checked_outer_seed = _checked_outer_seed(outer_seed_u64)
    pure = _baseline_slots(BfclV4PilotArm.PURE, "bare", checked_outer_seed)
    static = _baseline_slots(BfclV4PilotArm.STATIC, "static-frozen", checked_outer_seed)
    score = _adaptive_slots(BfclV4PilotArm.SCORE, checked_outer_seed)
    full = _adaptive_slots(BfclV4PilotArm.FULL, checked_outer_seed)
    pure_at_b = _pure_at_b_slots(checked_outer_seed)
    topological = (
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
    calls = tuple(
        BfclV4PilotCallSlot.model_validate(
            {**item.model_dump(mode="python"), "global_slot": index},
            strict=True,
        )
        for index, item in enumerate(topological)
    )
    return BfclV4PublicPilotCallPlan(
        manifest_fingerprint=BFCL_V4_PUBLIC_PILOT_MANIFEST.fingerprint,
        outer_seed_u64=checked_outer_seed,
        calls=calls,
        schedule_content_sha256=bfcl_v4_pilot_schedule_content_sha256(
            calls,
            outer_seed_u64=checked_outer_seed,
        ),
    )


__all__ = ["build_bfcl_v4_public_pilot_call_plan"]
