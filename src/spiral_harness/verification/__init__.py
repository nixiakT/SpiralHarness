"""Independent statistics and promotion gate for SpiralHarness."""

from spiral_harness.verification.gate import PromotionGate, evaluate_gate
from spiral_harness.verification.models import (
    ComparisonMetrics,
    ComparisonResult,
    ConfidenceInterval,
    Decision,
    GateCheck,
    GateCheckOutcome,
    GateConfig,
    GateDecision,
    MechanismCheck,
    MechanismEvidence,
    PairingAudit,
    SliceMetrics,
    TaskComparison,
    TrialObservation,
    TrialStatus,
)
from spiral_harness.verification.statistics import (
    compare_trials,
    compute_paired_statistics,
    paired_bootstrap_ci,
    paired_comparison,
)

__all__ = [
    "ComparisonMetrics",
    "ComparisonResult",
    "ConfidenceInterval",
    "Decision",
    "GateCheck",
    "GateCheckOutcome",
    "GateConfig",
    "GateDecision",
    "MechanismCheck",
    "MechanismEvidence",
    "PairingAudit",
    "PromotionGate",
    "SliceMetrics",
    "TaskComparison",
    "TrialObservation",
    "TrialStatus",
    "compare_trials",
    "compute_paired_statistics",
    "evaluate_gate",
    "paired_bootstrap_ci",
    "paired_comparison",
]
