from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.verification import (
    Decision,
    GateCheckOutcome,
    GateConfig,
    MechanismCheck,
    MechanismEvidence,
    PromotionGate,
    TrialObservation,
    TrialStatus,
)


def paired_trials(
    deltas: list[float],
    *,
    protected_tasks: set[int] | None = None,
    candidate_tokens: int = 100,
    candidate_latency_ms: float = 50,
    candidate_tool_calls: int = 2,
) -> tuple[list[TrialObservation], list[TrialObservation]]:
    protected_tasks = protected_tasks or set()
    parent: list[TrialObservation] = []
    candidate: list[TrialObservation] = []
    for index, delta in enumerate(deltas):
        task_id = f"task-{index:02d}"
        tags = ("protected",) if index in protected_tasks else ()
        common = {
            "task_id": task_id,
            "seed": 11,
            "slice_tags": tags,
            "execution_fingerprint": f"fixed-runtime:{task_id}:11",
        }
        parent.append(
            TrialObservation(
                harness_id="parent",
                score=0.5,
                tokens=100,
                latency_ms=50,
                tool_calls=2,
                **common,
            )
        )
        candidate.append(
            TrialObservation(
                harness_id="candidate",
                score=0.5 + delta,
                tokens=candidate_tokens,
                latency_ms=candidate_latency_ms,
                tool_calls=candidate_tool_calls,
                **common,
            )
        )
    return parent, candidate


def config(**updates: object) -> GateConfig:
    defaults: dict[str, object] = {
        "min_tasks": 5,
        "min_effect": 0.04,
        "bootstrap_samples": 1_000,
        "bootstrap_seed": 123,
    }
    defaults.update(updates)
    return GateConfig(**defaults)


def checks_by_name(decision) -> dict[str, object]:
    return {check.name: check for check in decision.checks}


def test_stable_improvement_is_promoted_with_full_audit() -> None:
    parent, candidate = paired_trials([0.1] * 8)

    result = PromotionGate(config()).evaluate(parent, candidate)

    assert result.decision is Decision.PROMOTE
    assert result.metrics is not None
    assert result.metrics.n_tasks == 8
    assert result.metrics.confidence_interval.lower == pytest.approx(0.1)
    assert all(check.outcome is GateCheckOutcome.PASS for check in result.checks)
    assert result.reasons == ("all configured promotion checks passed",)


def test_noisy_apparent_gain_is_inconclusive() -> None:
    parent, candidate = paired_trials([0.2, -0.1, 0.2, -0.1, 0.2, -0.1])

    result = PromotionGate(config()).evaluate(parent, candidate)

    assert result.decision is Decision.INCONCLUSIVE
    primary = checks_by_name(result)["primary_effect"]
    assert primary.outcome is GateCheckOutcome.INCONCLUSIVE
    assert result.metrics is not None
    assert result.metrics.mean_delta == pytest.approx(0.05)
    assert result.metrics.confidence_interval.lower <= 0


def test_real_degradation_is_rejected() -> None:
    parent, candidate = paired_trials([-0.1] * 6)

    result = PromotionGate(config()).evaluate(parent, candidate)

    assert result.decision is Decision.REJECT
    assert checks_by_name(result)["primary_effect"].outcome is GateCheckOutcome.FAIL


def test_protected_slice_regression_rejects_a_global_gain() -> None:
    parent, candidate = paired_trials(
        [-0.1, -0.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        protected_tasks={0, 1},
    )
    gate_config = config(protected_slice_floors={"protected": -0.05})

    result = PromotionGate(gate_config).evaluate(parent, candidate)

    assert result.metrics is not None
    assert result.metrics.mean_delta > 0
    assert result.decision is Decision.REJECT
    slice_check = checks_by_name(result)["protected_slices"]
    assert slice_check.outcome is GateCheckOutcome.FAIL
    assert "below floor" in " ".join(slice_check.reasons)


def test_protected_slice_lower_bound_must_strictly_clear_floor() -> None:
    parent, candidate = paired_trials(
        [-0.05, -0.05, *([0.3] * 8)],
        protected_tasks={0, 1},
    )

    result = PromotionGate(config(protected_slice_floors={"protected": -0.05})).evaluate(
        parent, candidate
    )

    assert result.decision is Decision.INCONCLUSIVE
    assert checks_by_name(result)["protected_slices"].outcome is GateCheckOutcome.INCONCLUSIVE


def test_resource_ratio_is_a_hard_constraint_and_beats_uncertainty() -> None:
    parent, candidate = paired_trials([0.1] * 6, candidate_tokens=130)
    # The deliberately impossible sample target also makes sample_size
    # inconclusive; a simultaneous hard cost failure must still reject.
    gate_config = config(min_tasks=20, max_tokens_ratio=1.2)

    result = PromotionGate(gate_config).evaluate(parent, candidate)

    assert result.decision is Decision.REJECT
    assert checks_by_name(result)["sample_size"].outcome is GateCheckOutcome.INCONCLUSIVE
    resources = checks_by_name(result)["resources"]
    assert resources.outcome is GateCheckOutcome.FAIL
    assert resources.metrics["tokens"]["ratio"] == pytest.approx(1.3)


def test_fingerprint_mismatch_is_rejected_not_silently_dropped() -> None:
    parent, candidate = paired_trials([0.1] * 6)
    candidate[0] = candidate[0].model_copy(update={"execution_fingerprint": "different"})

    result = PromotionGate(config()).evaluate(parent, candidate)

    assert result.decision is Decision.REJECT
    integrity = checks_by_name(result)["integrity"]
    assert integrity.outcome is GateCheckOutcome.FAIL
    assert integrity.metrics["fingerprint_mismatches"] == ["task-00::seed=11"]


def test_duplicate_pair_is_rejected_even_if_other_tasks_improve() -> None:
    parent, candidate = paired_trials([0.1] * 6)
    candidate.append(candidate[0].model_copy(update={"score": 1.0}))

    result = PromotionGate(config()).evaluate(parent, candidate)

    assert result.decision is Decision.REJECT
    integrity = checks_by_name(result)["integrity"]
    assert integrity.outcome is GateCheckOutcome.FAIL
    assert integrity.metrics["duplicate_candidate_pairs"] == ["task-00::seed=11"]


def test_missing_mechanism_check_is_inconclusive_but_failed_check_rejects() -> None:
    parent, candidate = paired_trials([0.1] * 6)
    gate = PromotionGate(config(required_mechanism_checks=("activation", "adherence")))

    missing = gate.evaluate(
        parent,
        candidate,
        MechanismEvidence(
            candidate_harness_id="candidate",
            checks=(
                MechanismCheck(
                    name="activation",
                    passed=True,
                    evidence_refs=("activation-evidence",),
                ),
            ),
        ),
    )
    failed = gate.evaluate(
        parent,
        candidate,
        MechanismEvidence(
            candidate_harness_id="candidate",
            checks=(
                MechanismCheck(
                    name="activation",
                    passed=True,
                    evidence_refs=("activation-evidence",),
                ),
                MechanismCheck(name="adherence", passed=False, details="probe contradicted patch"),
            ),
        ),
    )

    assert missing.decision is Decision.INCONCLUSIVE
    assert checks_by_name(missing)["mechanism"].outcome is GateCheckOutcome.INCONCLUSIVE
    assert failed.decision is Decision.REJECT
    assert checks_by_name(failed)["mechanism"].outcome is GateCheckOutcome.FAIL


def test_all_required_mechanism_checks_allow_promotion() -> None:
    parent, candidate = paired_trials([0.1] * 6)
    gate = PromotionGate(config(required_mechanism_checks=("activation", "adherence")))
    evidence = MechanismEvidence(
        candidate_harness_id="candidate",
        checks=(
            MechanismCheck(name="activation", passed=True, evidence_refs=("activation-evidence",)),
            MechanismCheck(name="adherence", passed=True, evidence_refs=("adherence-evidence",)),
        ),
    )

    result = gate.evaluate(parent, candidate, evidence)

    assert result.decision is Decision.PROMOTE


def test_policy_violation_is_always_rejected() -> None:
    parent, candidate = paired_trials([0.1] * 6)
    candidate[0] = candidate[0].model_copy(
        update={"status": TrialStatus.POLICY_VIOLATION, "violations": ("forbidden read",)}
    )

    result = PromotionGate(config()).evaluate(parent, candidate)

    assert result.decision is Decision.REJECT
    assert checks_by_name(result)["policy"].outcome is GateCheckOutcome.FAIL


def test_optional_single_task_and_regression_rate_constraints() -> None:
    parent, candidate = paired_trials([-0.15, *([0.2] * 9)])
    gate_config = config(max_single_task_regression=0.1, max_regression_rate=0.05)

    result = PromotionGate(gate_config).evaluate(parent, candidate)

    assert result.decision is Decision.REJECT
    regressions = checks_by_name(result)["task_regressions"]
    assert regressions.outcome is GateCheckOutcome.FAIL
    assert regressions.metrics["regression_rate"] == pytest.approx(0.1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_protected_slice_floors_must_be_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        GateConfig(protected_slice_floors={"protected": value})


@pytest.mark.parametrize("value", ["yes", 1])
def test_mechanism_passed_is_a_strict_boolean(value: object) -> None:
    with pytest.raises(ValidationError):
        MechanismCheck(
            name="activation",
            passed=value,
            evidence_refs=("activation-evidence",),
        )


def test_passed_mechanism_check_requires_unique_evidence_refs() -> None:
    with pytest.raises(ValidationError, match="requires evidence_refs"):
        MechanismCheck(name="activation", passed=True)

    with pytest.raises(ValidationError, match="must be unique"):
        MechanismCheck(
            name="activation",
            passed=True,
            evidence_refs=("same", "same"),
        )


def test_required_mechanism_evidence_must_bind_exact_candidate() -> None:
    parent, candidate = paired_trials([0.1] * 6)
    gate = PromotionGate(config(required_mechanism_checks=("activation",)))
    check = MechanismCheck(
        name="activation",
        passed=True,
        evidence_refs=("activation-evidence",),
    )

    unbound = gate.evaluate(parent, candidate, MechanismEvidence(checks=(check,)))
    wrong = gate.evaluate(
        parent,
        candidate,
        MechanismEvidence(candidate_harness_id="other-candidate", checks=(check,)),
    )

    assert unbound.decision is Decision.REJECT
    assert wrong.decision is Decision.REJECT
    assert checks_by_name(unbound)["mechanism"].outcome is GateCheckOutcome.FAIL
    assert checks_by_name(wrong)["mechanism"].outcome is GateCheckOutcome.FAIL


def test_raw_mechanism_mapping_is_not_trusted_evidence() -> None:
    parent, candidate = paired_trials([0.1] * 6)
    gate = PromotionGate(config(required_mechanism_checks=("activation",)))

    with pytest.raises(TypeError, match="raw mechanism mappings"):
        gate.evaluate(parent, candidate, {"activation": True})  # type: ignore[arg-type]


def test_gate_uses_independent_config_snapshot() -> None:
    parent, candidate = paired_trials(
        [-0.2, *([0.5] * 9)],
        protected_tasks={0},
    )
    gate_config = config(protected_slice_floors={"protected": 0.0})
    gate = PromotionGate(gate_config)

    before = gate.evaluate(parent, candidate)
    gate_config.protected_slice_floors.clear()
    exposed_copy = gate.config
    exposed_copy.protected_slice_floors.clear()
    after = gate.evaluate(parent, candidate)

    assert before.decision is Decision.REJECT
    assert after.decision is Decision.REJECT
    assert before.gate_config_sha256 == after.gate_config_sha256
    assert gate.config.protected_slice_floors == {"protected": 0.0}


def test_unregistered_seeds_are_hard_failures_and_do_not_change_metrics() -> None:
    parent, candidate = paired_trials([-0.1] * 5)
    registered_task_ids = tuple(trial.task_id for trial in parent)
    for parent_trial, candidate_trial in zip(tuple(parent), tuple(candidate), strict=True):
        fingerprint = f"fixed-runtime:{parent_trial.task_id}:12"
        parent.append(
            parent_trial.model_copy(update={"seed": 12, "execution_fingerprint": fingerprint})
        )
        candidate.append(
            candidate_trial.model_copy(
                update={"seed": 12, "score": 1.5, "execution_fingerprint": fingerprint}
            )
        )

    result = PromotionGate(
        config(
            expected_task_ids=registered_task_ids,
            expected_seeds=(11,),
            min_effect=0.0,
        )
    ).evaluate(parent, candidate)

    assert result.decision is Decision.REJECT
    assert result.metrics is not None
    assert result.metrics.n_valid_pairs == 5
    assert result.metrics.mean_delta == pytest.approx(-0.1)
    assert len(result.comparison.audit.unexpected_parent_pairs) == 5
    assert len(result.comparison.audit.unexpected_candidate_pairs) == 5
    assert checks_by_name(result)["integrity"].outcome is GateCheckOutcome.FAIL


def test_parent_and_candidate_must_have_distinct_harness_ids() -> None:
    parent, candidate = paired_trials([0.1] * 6)
    candidate = [trial.model_copy(update={"harness_id": "parent"}) for trial in candidate]

    result = PromotionGate(config()).evaluate(parent, candidate)

    assert result.decision is Decision.REJECT
    assert "same harness ID" in " ".join(result.comparison.audit.integrity_errors)


def test_gate_revalidates_unchecked_observation_copies() -> None:
    parent, candidate = paired_trials([0.1] * 6)
    candidate[0] = candidate[0].model_copy(update={"tokens": float("nan")})

    with pytest.raises(ValidationError):
        PromotionGate(config(max_tokens_ratio=1.2)).evaluate(parent, candidate)


def test_zero_over_zero_resource_use_satisfies_ratio_cap() -> None:
    parent, candidate = paired_trials([0.1] * 6, candidate_tool_calls=0)
    parent = [trial.model_copy(update={"tool_calls": 0}) for trial in parent]

    result = PromotionGate(config(max_tool_calls_ratio=0.5)).evaluate(parent, candidate)

    assert result.decision is Decision.PROMOTE
    resources = checks_by_name(result)["resources"]
    assert resources.outcome is GateCheckOutcome.PASS
    assert resources.metrics["tool_calls"]["ratio"] == 0.0


def test_gate_config_rejects_boolean_integer_coercion() -> None:
    with pytest.raises(ValidationError):
        GateConfig(min_tasks=True)
