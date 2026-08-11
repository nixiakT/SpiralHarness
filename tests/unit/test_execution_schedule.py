from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    AttemptBudget,
    AttemptLedgerState,
)
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    EvaluationCellKey,
    EvaluationPhase,
    EvaluationSide,
    ScheduleBudgetExceeded,
    derive_seed_v2,
    preflight_attempt_budget,
    publish_schedule_preflight,
)
from spiral_harness.storage import ArtifactStore


def batch(**updates: object) -> EvaluationBatchSchedule:
    values: dict[str, object] = {
        "study": "gsm8k-prompt-evolution-v1",
        "kind": "evidence-targeted",
        "phase": EvaluationPhase.GATE,
        "query": 3,
        "master_seed": 20260811,
        "parent_harness_id": "champion@sha256:parent",
        "candidate_harness_id": "candidate@sha256:child",
        "task_ids": ("task-002", "task-001"),
        "search_runs": (103, 101),
        "repeat_seeds": (17, 11),
        "max_attempts_per_cell": 2,
        "token_ceiling_per_attempt": 10,
    }
    values.update(updates)
    return EvaluationBatchSchedule(**values)


def exact_budget(schedule: EvaluationBatchSchedule) -> AttemptBudget:
    return AttemptBudget(
        max_attempts=schedule.required_attempts,
        max_total_tokens=schedule.required_tokens,
        max_tokens_per_attempt=schedule.token_ceiling_per_attempt,
    )


def test_complete_cell_and_paired_key_keep_adaptive_coordinates() -> None:
    schedule = batch()
    parent = next(schedule.iter_cells())
    candidate = parent.model_copy(update={"side": EvaluationSide.CANDIDATE})

    assert parent.side is EvaluationSide.PARENT
    assert parent.fingerprint != candidate.fingerprint
    assert parent.pairing_fingerprint == candidate.pairing_fingerprint
    assert parent.paired_key.search_run == 101
    assert parent.paired_key.phase is EvaluationPhase.GATE
    assert parent.paired_key.query == 3

    for update in (
        {"search_run": 107},
        {"phase": EvaluationPhase.EXPLORATION},
        {"query": 4},
        {"task_id": "task-999"},
        {"repeat_seed": 19},
    ):
        changed = EvaluationCellKey.model_validate(
            parent.model_copy(update=update),
            strict=True,
        )
        assert changed.pairing_fingerprint != parent.pairing_fingerprint

    with pytest.raises(ValidationError):
        parent.query = 99
    with pytest.raises(ValidationError, match="exact and non-empty"):
        EvaluationCellKey(
            study=" study ",
            kind="static",
            search_run=1,
            phase=EvaluationPhase.GATE,
            query=0,
            task_id="task",
            repeat_seed=2,
            side=EvaluationSide.PARENT,
        )


def test_schedule_is_canonical_lazy_complete_and_binds_side_harnesses() -> None:
    schedule = batch()
    cells = tuple(schedule.iter_cells())

    assert schedule.task_ids == ("task-001", "task-002")
    assert schedule.search_runs == (101, 103)
    assert schedule.repeat_seeds == (11, 17)
    assert schedule.sides == (EvaluationSide.PARENT, EvaluationSide.CANDIDATE)
    assert schedule.cell_count == len(cells) == 16
    assert schedule.required_attempts == 32
    assert schedule.required_tokens == 320
    assert len({cell.fingerprint for cell in cells}) == 16
    assert all(schedule.contains(cell) for cell in cells)
    assert schedule.harness_id_for(EvaluationSide.PARENT) == schedule.parent_harness_id
    assert schedule.harness_id_for(EvaluationSide.CANDIDATE) == schedule.candidate_harness_id

    static = batch(
        parent_harness_id="same",
        candidate_harness_id="same",
    )
    assert static.harness_id_for(EvaluationSide.PARENT) == static.harness_id_for(
        EvaluationSide.CANDIDATE
    )

    with pytest.raises(ValidationError, match="duplicates"):
        batch(task_ids=("task-1", "task-1"))
    with pytest.raises(ValidationError, match="exactly parent and candidate"):
        batch(sides=(EvaluationSide.PARENT,))


def test_seed_v2_is_domain_separated_deterministic_and_paired() -> None:
    schedule = batch()
    parent = next(schedule.iter_cells())
    candidate = parent.model_copy(update={"side": EvaluationSide.CANDIDATE})

    parent_seed = schedule.seed_for(parent)
    assert parent_seed == schedule.seed_for(parent)
    assert parent_seed == schedule.seed_for(candidate)
    assert 0 <= parent_seed < 2**63
    assert schedule.seed_for(parent, attempt_index=1) != parent_seed
    assert (
        derive_seed_v2(
            master_seed=schedule.master_seed,
            domain="proposal",
            coordinates=parent.paired_key,
        )
        != parent_seed
    )
    assert derive_seed_v2(
        master_seed=schedule.master_seed,
        domain="model-rollout",
        coordinates=parent,
    ) != derive_seed_v2(
        master_seed=schedule.master_seed,
        domain="model-rollout",
        coordinates=candidate,
    )

    with pytest.raises(TypeError, match="master_seed"):
        derive_seed_v2(master_seed=True, domain="x", coordinates=parent)
    with pytest.raises(ValueError, match="domain"):
        derive_seed_v2(master_seed=1, domain=" x ", coordinates=parent)
    with pytest.raises(ValueError, match="retry ceiling"):
        schedule.seed_for(parent, attempt_index=2)


def test_all_or_nothing_preflight_accepts_exact_capacity_and_rejects_difference_of_one() -> None:
    schedule = batch()
    certificate = preflight_attempt_budget(schedule, exact_budget(schedule))

    assert certificate.schedule_fingerprint == schedule.fingerprint
    assert certificate.cell_count == 16
    assert certificate.required_attempts == certificate.available_attempts == 32
    assert certificate.required_tokens == certificate.available_tokens == 320
    assert certificate.ledger_id is None
    assert certificate.ledger_tail_ref is None

    with pytest.raises(ScheduleBudgetExceeded, match="attempt budget"):
        preflight_attempt_budget(
            schedule,
            AttemptBudget(
                max_attempts=schedule.required_attempts - 1,
                max_total_tokens=schedule.required_tokens,
                max_tokens_per_attempt=schedule.token_ceiling_per_attempt,
            ),
        )
    with pytest.raises(ScheduleBudgetExceeded, match="token budget"):
        preflight_attempt_budget(
            schedule,
            AttemptBudget(
                max_attempts=schedule.required_attempts,
                max_total_tokens=schedule.required_tokens - 1,
                max_tokens_per_attempt=schedule.token_ceiling_per_attempt,
            ),
        )
    with pytest.raises(ScheduleBudgetExceeded, match="per-attempt ceiling"):
        preflight_attempt_budget(
            schedule,
            AttemptBudget(
                max_attempts=schedule.required_attempts,
                max_total_tokens=schedule.required_tokens,
                max_tokens_per_attempt=schedule.token_ceiling_per_attempt - 1,
            ),
        )


def test_preflight_binds_exact_clean_ledger_tail_and_is_persisted(tmp_path) -> None:
    schedule = batch()
    budget = AttemptBudget(
        max_attempts=40,
        max_total_tokens=400,
        max_tokens_per_attempt=10,
    )
    tail = ArtifactRef(
        sha256="a" * 64,
        size=123,
        media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
    )
    state = AttemptLedgerState(
        ledger_id="study-ledger",
        budget=budget,
        tail_ref=tail,
        pending_reservation_ref=None,
        poisoned=False,
        attempts_used=8,
        completed_attempts=8,
        charged_tokens=80,
        encumbered_tokens=80,
        remaining_attempts=32,
        remaining_tokens=320,
    )

    certificate = preflight_attempt_budget(schedule, state)
    assert certificate.ledger_id == "study-ledger"
    assert certificate.ledger_tail_ref == tail
    store = ArtifactStore(tmp_path / "cas")
    ref = publish_schedule_preflight(store, certificate)
    assert ref.media_type == SCHEDULE_PREFLIGHT_MEDIA_TYPE
    assert ref.sha256 == certificate.fingerprint
    assert store.get_json(ref, type(certificate)) == certificate

    pending = state.model_copy(
        update={
            "tail_ref": ArtifactRef(
                sha256="b" * 64,
                size=1,
                media_type="application/vnd.spiral-harness.attempt-reservation.v1+json",
            ),
            "pending_reservation_ref": ArtifactRef(
                sha256="b" * 64,
                size=1,
                media_type="application/vnd.spiral-harness.attempt-reservation.v1+json",
            ),
        }
    )
    with pytest.raises(ScheduleBudgetExceeded, match="open reservation"):
        preflight_attempt_budget(schedule, pending)
