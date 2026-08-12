from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from test_terminal_decision import build_graph

from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
)
from spiral_harness.core.models import ArtifactRef, BudgetPolicy
from spiral_harness.execution.schedule import SCHEDULE_PREFLIGHT_MEDIA_TYPE
from spiral_harness.experiments.controller_artifacts import (
    EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
    EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE,
    SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
    SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
    ExperimentUsageClaim,
    ExperimentUsageEntry,
    ExperimentUsageEntryV1,
    SkillProbeSettlementKind,
    SkillProbeUsageArmSettlement,
    SkillProbeUsageClaim,
    SkillProbeUsageSettlementClaim,
)
from spiral_harness.experiments.experiment_usage import (
    ExperimentUsageLedger,
    ExperimentUsageLedgerError,
)
from spiral_harness.experiments.skill_probe_authorization import (
    SkillProbeExecutionAuthorization,
)
from spiral_harness.experiments.skill_probe_closure import MatchedSkillProbeClosure
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.storage.journal import JOURNAL_ENTRY_MEDIA_TYPE
from tests.integration.test_matched_skill_probe_execution import _run
from tests.unit.test_skill_probe_preregistration import (
    _controller_at_probes,
    _probe_fixture,
)


def _replace[ModelT: BaseModel](model: ModelT, **updates: object) -> ModelT:
    values = model.model_dump(mode="python", round_trip=True, warnings="none")
    values.update(updates)
    return type(model).model_validate(values, strict=True)


def _ref(digit: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=1, media_type=media_type)


def test_frozen_v1_rejects_probe_claim_nullable_wall_time_and_v2_parent() -> None:
    values = {
        "experiment_ref": _ref(
            "1",
            "application/vnd.spiral-harness.experiment-manifest.v1+json",
        ),
        "protocol_ref": _ref(
            "2",
            "application/vnd.spiral-harness.protocol-manifest.v1+json",
        ),
        "sequence": 0,
        "claim_ref": _ref("3", SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE),
        "cumulative_evaluations": 1,
        "cumulative_tokens": 1,
        "cumulative_tool_calls": 0,
        "cumulative_wall_time_seconds": 0.0,
        "cumulative_cost_usd": None,
        "previous_entry_ref": None,
    }

    with pytest.raises(ValidationError, match="v1 claim_ref"):
        ExperimentUsageEntryV1.model_validate(values, strict=True)
    values.update(
        claim_ref=_ref("4", EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE),
        cumulative_wall_time_seconds=None,
    )
    with pytest.raises(ValidationError, match="valid number"):
        ExperimentUsageEntryV1.model_validate(values, strict=True)
    values.update(
        sequence=1,
        cumulative_wall_time_seconds=0.0,
        previous_entry_ref=_ref("5", EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE),
    )
    with pytest.raises(ValidationError, match="v1 previous_entry_ref"):
        ExperimentUsageEntryV1.model_validate(values, strict=True)


def test_real_v1_gate_tail_appends_and_replays_a_v2_probe_reservation(
    tmp_path: Path,
) -> None:
    probe = _probe_fixture(tmp_path / "probe")
    gate = build_graph(tmp_path / "gate")
    controller, _, probes = _controller_at_probes(probe)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=probe.candidate_ref,
        running_probes_tail_ref=probes,
    )
    authorization = probe.store.get_json(
        authorization_ref,
        SkillProbeExecutionAuthorization,
    )
    evaluation = gate.evaluation.model_copy(update={"candidate_ref": probe.candidate_ref})
    tokens, tools, wall_time, cost = ExperimentUsageLedger.resource_charge(
        gate.parent_batch,
        gate.candidate_batch,
    )
    units = len(gate.parent_batch.observations) + len(gate.candidate_batch.observations)
    gate_claim = ExperimentUsageClaim(
        experiment_ref=probe.experiment_ref,
        protocol_ref=probe.protocol_ref,
        candidate_ref=probe.candidate_ref,
        running_gate_tail_ref=probes,
        evaluation_ref=gate.evaluation_ref,
        parent_batch_ref=gate.parent_batch_ref,
        candidate_batch_ref=gate.candidate_batch_ref,
        evaluation_units=units,
        tokens=tokens,
        tool_calls=tools,
        wall_time_seconds=wall_time,
        cost_usd=cost,
    )
    gate_claim_ref = probe.store.put_json(
        gate_claim,
        media_type=EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
    )
    v1_tail = probe.store.put_json(
        ExperimentUsageEntryV1(
            experiment_ref=probe.experiment_ref,
            protocol_ref=probe.protocol_ref,
            sequence=0,
            claim_ref=gate_claim_ref,
            cumulative_evaluations=units,
            cumulative_tokens=tokens,
            cumulative_tool_calls=tools,
            cumulative_wall_time_seconds=wall_time,
            cumulative_cost_usd=cost,
            previous_entry_ref=None,
        ),
        media_type=EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE,
    )

    def verify_gate_history(_tail: ArtifactRef):
        return probe.candidate, probe.plan_ref, probe.plan_ref

    def verify_gate_evaluation(**_kwargs: object):
        return evaluation, gate.parent_batch, gate.candidate_batch, units

    ledger = ExperimentUsageLedger(
        probe.store,
        experiment_ref=probe.experiment_ref,
        protocol_ref=probe.protocol_ref,
        budget_limits=BudgetPolicy(max_evaluations=100),
        verify_gate_history=verify_gate_history,
        verify_gate_evaluation=verify_gate_evaluation,
    )
    reservation_tail = ledger.reserve_skill_probe(
        tail_ref=v1_tail,
        authorization_ref=authorization_ref,
        authorization=authorization,
        plan=probe.plan,
    )
    usage = ledger.replay(reservation_tail)
    reservation_entry = probe.store.get_json(reservation_tail, ExperimentUsageEntry)

    assert reservation_tail.media_type == EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE
    assert reservation_entry.schema_version == "2"
    assert reservation_entry.previous_entry_ref == v1_tail
    assert usage.entry_refs == (v1_tail, reservation_tail)
    assert usage.claim_refs == (gate_claim_ref,)
    assert usage.skill_probe_authorization_refs == (authorization_ref,)


def test_completed_settlement_replays_exact_closure_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    executable, controller, execution = _run(tmp_path)
    closure = executable.graph.store.get_json(
        execution.result.closure_ref,
        MatchedSkillProbeClosure,
    )
    settle = {
        "tail_ref": controller.usage_tail_ref,
        "authorization_ref": closure.authorization_ref,
        "revert_preflight_ref": closure.revert.preflight_ref,
        "revert_terminal_tail_ref": closure.revert.closing_ledger_tail_ref,
        "placebo_preflight_ref": closure.placebo.preflight_ref,
        "placebo_terminal_tail_ref": closure.placebo.closing_ledger_tail_ref,
        "terminal_kind": SkillProbeSettlementKind.COMPLETED,
        "closure_ref": execution.result.closure_ref,
    }
    settled_tail = controller._usage_ledger.settle_skill_probe(**settle)
    settled_usage = controller._usage_ledger.replay(settled_tail)
    assert settled_usage.skill_probe_settlement_refs
    assert (
        controller._usage_ledger.settle_skill_probe(**{**settle, "tail_ref": settled_tail})
        == settled_tail
    )

    settlement_ref = settled_usage.skill_probe_settlement_refs[0]
    settlement = executable.graph.store.get_json(
        settlement_ref,
        SkillProbeUsageSettlementClaim,
    )
    forged_closure = _replace(
        closure,
        running_probes_tail_ref=_ref("7", JOURNAL_ENTRY_MEDIA_TYPE),
    )
    forged_closure_ref = executable.graph.store.put_json(
        forged_closure,
        media_type=execution.result.closure_ref.media_type,
    )
    forged_claim_ref = executable.graph.store.put_json(
        _replace(settlement, closure_ref=forged_closure_ref),
        media_type=SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
    )
    settled_entry = executable.graph.store.get_json(settled_tail, ExperimentUsageEntry)
    forged_tail = executable.graph.store.put_json(
        _replace(settled_entry, claim_ref=forged_claim_ref),
        media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    )
    with pytest.raises(ExperimentUsageLedgerError, match="reservation context"):
        controller._usage_ledger.replay(forged_tail)


def test_poisoned_settlement_is_a_terminal_usage_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    experiment_ref = _ref(
        "1",
        EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )
    protocol_ref = _ref(
        "2",
        PROTOCOL_MANIFEST_MEDIA_TYPE,
    )
    authorization_ref = _ref(
        "3",
        "application/vnd.spiral-harness.skill-probe-execution-authorization.v1+json",
    )
    reservation = SkillProbeUsageClaim(
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        candidate_ref=_ref(
            "4",
            CANDIDATE_MANIFEST_MEDIA_TYPE,
        ),
        authorization_ref=authorization_ref,
        execution_nonce="5" * 64,
        plan_ref=_ref(
            "6",
            "application/vnd.spiral-harness.skill-mechanism-plan.v1+json",
        ),
        running_probes_tail_ref=_ref("7", JOURNAL_ENTRY_MEDIA_TYPE),
        revert_schedule_fingerprint="8" * 64,
        placebo_schedule_fingerprint="9" * 64,
        evaluation_units=1,
        tokens=1,
    )
    reservation_ref = store.put_json(
        reservation,
        media_type=SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
    )
    reservation_tail = store.put_json(
        ExperimentUsageEntry(
            experiment_ref=experiment_ref,
            protocol_ref=protocol_ref,
            sequence=0,
            claim_ref=reservation_ref,
            cumulative_evaluations=1,
            cumulative_tokens=1,
            cumulative_tool_calls=0,
            cumulative_wall_time_seconds=None,
            cumulative_cost_usd=None,
            previous_entry_ref=None,
        ),
        media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    )
    settlement = SkillProbeUsageSettlementClaim(
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        candidate_ref=reservation.candidate_ref,
        authorization_ref=authorization_ref,
        execution_nonce=reservation.execution_nonce,
        reservation_claim_ref=reservation_ref,
        terminal_kind=SkillProbeSettlementKind.FAILED,
        revert=SkillProbeUsageArmSettlement(
            control="revert",
            preflight_ref=_ref(
                "a",
                SCHEDULE_PREFLIGHT_MEDIA_TYPE,
            ),
            terminal_tail_ref=None,
            encumbered_tokens=2,
            poisoned=True,
        ),
        placebo=SkillProbeUsageArmSettlement(
            control="placebo",
            preflight_ref=_ref(
                "b",
                SCHEDULE_PREFLIGHT_MEDIA_TYPE,
            ),
            terminal_tail_ref=None,
            encumbered_tokens=0,
            poisoned=False,
        ),
        reserved_tokens=1,
        encumbered_tokens=2,
        token_adjustment=1,
        poisoned=True,
    )
    settlement_ref = store.put_json(
        settlement,
        media_type=SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
    )
    settlement_tail = store.put_json(
        ExperimentUsageEntry(
            experiment_ref=experiment_ref,
            protocol_ref=protocol_ref,
            sequence=1,
            claim_ref=settlement_ref,
            cumulative_evaluations=1,
            cumulative_tokens=2,
            cumulative_tool_calls=0,
            cumulative_wall_time_seconds=None,
            cumulative_cost_usd=None,
            poisoned=True,
            previous_entry_ref=reservation_tail,
        ),
        media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    )
    appended_tail = store.put_json(
        ExperimentUsageEntry(
            experiment_ref=experiment_ref,
            protocol_ref=protocol_ref,
            sequence=2,
            claim_ref=reservation_ref,
            cumulative_evaluations=1,
            cumulative_tokens=2,
            cumulative_tool_calls=0,
            cumulative_wall_time_seconds=None,
            cumulative_cost_usd=None,
            poisoned=True,
            previous_entry_ref=settlement_tail,
        ),
        media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    )
    ledger = ExperimentUsageLedger(
        store,
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        budget_limits=BudgetPolicy(max_evaluations=1, max_tokens=1),
        verify_gate_history=lambda _ref: (_ for _ in ()).throw(AssertionError),
        verify_gate_evaluation=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr(
        ledger._probe_usage,
        "replay_reservation",
        lambda *_args, **_kwargs: (1, 1, 0, None, None),
    )
    monkeypatch.setattr(
        ledger._probe_usage,
        "replay_settlement",
        lambda *_args, **_kwargs: ((0, 1, 0, 0.0, 0.0), True),
    )

    poisoned = ledger.replay(settlement_tail)
    assert poisoned.poisoned is True
    assert poisoned.total_tokens == 2
    assert poisoned.remaining_tokens == 0
    with pytest.raises(ExperimentUsageLedgerError, match="poisoned terminal"):
        ledger.replay(appended_tail)
