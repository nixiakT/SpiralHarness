"""Shared deterministic helpers for model-driven HarnessFault development."""

from __future__ import annotations

from collections import defaultdict

from spiral_harness.benchmark._harness_fault_cases import (
    CANDIDATE_RULE_CATALOG,
    FaultFamily,
    HiddenScenarioSpec,
    RepairRuleId,
    RuntimeBranch,
    ScenarioRole,
    evaluate_branch,
)
from spiral_harness.benchmark.harness_fault import parse_harness_fault_output
from spiral_harness.benchmark.harness_fault_compiler import HarnessRole
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.execution.contracts import (
    CandidateTask,
    ExecutionStatus,
    FrozenModelSpec,
    ModelExecutionRecord,
)
from spiral_harness.experiments.confirmatory_arms import PromotionRule
from spiral_harness.experiments.harness_fault_development_contracts import (
    FaultDevelopmentGate,
    FaultDevelopmentObservation,
    FaultFeedbackItem,
    FeedbackEvidenceLinkage,
    HarnessFaultDevelopmentPlan,
    HarnessFaultFeedbackProjection,
    ModelAuthoredFaultProposal,
    PromotionDecision,
    ProposerVisibleFaultFeedback,
    ProposerVisibleFaultFeedbackItem,
    ProposerVisibleFeedbackMode,
)
from spiral_harness.storage.artifact_store import ArtifactStore

PROPOSER_TASK_ID = "harness-fault/model-driven-development/proposer"
PROPOSER_PLANE_VERSION = "spiral-harness.harness-fault-model-driven-proposer:v1"

_RULE_GUIDANCE = {
    RepairRuleId.CONSTANT_LEGACY: "legacy behavior on every family/context",
    RepairRuleId.CONSTANT_SAFE: "safe behavior on every family/context",
    RepairRuleId.ROUTED_POLICY: "route safe/legacy by the declared family and context",
    RepairRuleId.PROMPT_MEMORY_PATCH: "repair prompt and memory families in context-0",
    RepairRuleId.TOOL_PATCH: "repair both tool families in context-0",
    RepairRuleId.RUNTIME_PATCH: "repair middleware/control-flow/skill in context-0",
}

PROPOSER_SYSTEM_PROMPT = """You are the sole repair proposer for a public development run.
Choose exactly one rule from the finite catalog below. Do not rank several rules,
write a free-form patch, request another sample, or use human input. Return exactly
one compact JSON object of the form {\"rule_id\":\"<opaque-id>\"} and nothing else.

CATALOG
""" + "\n".join(f"{rule.value}: {_RULE_GUIDANCE[rule]}" for rule in CANDIDATE_RULE_CATALOG)


def solver_seed(master_seed: int, task_id: str) -> int:
    """Use the same paired seed for all attribution roles and factorial cells."""

    return int(
        canonical_sha256(
            {
                "schema": "spiral-harness/model-driven-fault-solver-seed/v1",
                "master_seed": master_seed,
                "task_id": task_id,
            }
        )[:8],
        16,
    )


def proposer_task(projection: HarnessFaultFeedbackProjection) -> CandidateTask:
    checked = HarnessFaultFeedbackProjection.model_validate(projection, strict=True)
    visible = proposer_visible_feedback(checked)
    packet = canonical_json_bytes(visible).decode("utf-8")
    question = (
        "Select one repair rule from the system catalog using this exact public-development "
        "feedback packet. Treatment assignment and authority-only source mapping are blinded.\n"
        f"FEEDBACK_SHA256={canonical_sha256(visible)}\n"
        f"FEEDBACK_JSON={packet}"
    )
    return CandidateTask(task_id=PROPOSER_TASK_ID, question=question)


def proposer_visible_feedback(
    projection: HarnessFaultFeedbackProjection,
) -> ProposerVisibleFaultFeedback:
    """Remove placebo labels and source IDs while retaining equal mechanism-view schema."""

    checked = HarnessFaultFeedbackProjection.model_validate(projection, strict=True)
    mode = (
        ProposerVisibleFeedbackMode.SCORE_ONLY
        if checked.linkage is FeedbackEvidenceLinkage.SCORE_ONLY
        else ProposerVisibleFeedbackMode.MECHANISM_VISIBLE
    )
    items = tuple(
        ProposerVisibleFaultFeedbackItem(
            task_id=item.task_id,
            family=item.family,
            surface=item.surface,
            scenario_role=item.scenario_role,
            raw_model_output_sha256=item.raw_model_output_sha256,
            parsed_output=item.parsed_output,
            inferred_branch=item.inferred_branch,
            behavior_correct=item.behavior_correct,
        )
        for item in checked.items
    )
    return ProposerVisibleFaultFeedback(
        feedback_mode=mode,
        parent_correct=checked.parent_correct,
        task_count=checked.task_count,
        items=items,
    )


def publish_proposer_harness(store: ArtifactStore, spec: FrozenModelSpec) -> ArtifactRef:
    """Publish the one frozen proposer prompt as a standard prompt-only harness."""

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    prompt_ref = store.put_bytes(PROPOSER_SYSTEM_PROMPT.encode("utf-8"), media_type="text/plain")
    manifest = HarnessManifest(
        model_fingerprint=checked_spec.model_fingerprint,
        runtime_fingerprint=checked_spec.runtime_fingerprint,
        trusted_plane_version=PROPOSER_PLANE_VERSION,
        components=(
            HarnessComponentRef(
                name="system-prompt",
                kind=ComponentKind.PROMPT,
                artifact=prompt_ref,
            ),
        ),
    )
    return store.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE)


def make_observation(
    *,
    scenario: HiddenScenarioSpec,
    role: HarnessRole,
    seed: int,
    record: ModelExecutionRecord,
) -> FaultDevelopmentObservation:
    """Grade the exact persisted model output without replacing or normalizing it."""

    execution = record.execution
    parsed = None
    inferred = None
    correct = False
    output_sha256 = None
    if execution.status is ExecutionStatus.COMPLETED:
        output = execution.output_text
        if output is None:  # pragma: no cover - ModelExecution invariant
            raise RuntimeError("completed execution omitted its raw output")
        output_sha256 = sha256_bytes(output.encode("utf-8"))
        try:
            parsed = parse_harness_fault_output(output)
        except ValueError:
            parsed = None
        if parsed is not None:
            safe = evaluate_branch(
                scenario.family,
                RuntimeBranch.SAFE,
                primary=scenario.primary,
                secondary=scenario.secondary,
            )
            legacy = evaluate_branch(
                scenario.family,
                RuntimeBranch.LEGACY,
                primary=scenario.primary,
                secondary=scenario.secondary,
            )
            actual = (parsed.answer, parsed.observable)
            if actual == safe:
                inferred = RuntimeBranch.SAFE
            elif actual == legacy:
                inferred = RuntimeBranch.LEGACY
            correct = actual == (scenario.expected_answer, scenario.expected_observable)
    return FaultDevelopmentObservation(
        harness_role=role,
        scenario_id=scenario.scenario_id,
        scenario_commitment=scenario.scenario_commitment,
        task_id=scenario.task.task_id,
        family=scenario.family,
        surface=scenario.surface,
        scenario_role=scenario.role,
        seed=seed,
        execution_status=execution.status,
        execution_ref=record.execution_ref,
        outcome_ref=record.outcome_ref,
        raw_model_output_sha256=output_sha256,
        parsed_output=parsed,
        inferred_branch=inferred,
        behavior_correct=correct,
    )


def build_feedback_projection(
    plan: HarnessFaultDevelopmentPlan,
    parent: tuple[FaultDevelopmentObservation, ...],
) -> HarnessFaultFeedbackProjection:
    """Project complete parent behavior as score, linked, or stratified placebo evidence."""

    checked_plan = HarnessFaultDevelopmentPlan.model_validate(plan, strict=True)
    return build_feedback_projection_from_coordinates(
        plan_fingerprint=checked_plan.fingerprint,
        feedback_linkage=checked_plan.feedback_linkage,
        task_count=checked_plan.task_count,
        parent=parent,
    )


def build_feedback_projection_from_coordinates(
    *,
    plan_fingerprint: str,
    feedback_linkage: FeedbackEvidenceLinkage,
    task_count: int,
    parent: tuple[FaultDevelopmentObservation, ...],
) -> HarnessFaultFeedbackProjection:
    """Project feedback from a truthful stage topology, without a synthetic run plan."""

    if type(feedback_linkage) is not FeedbackEvidenceLinkage:
        raise TypeError("feedback_linkage must be an exact FeedbackEvidenceLinkage")
    if type(task_count) is not int or task_count < 4:
        raise ValueError("task_count must be an integer of at least four")
    ordered = tuple(sorted(parent, key=lambda item: item.task_id))
    if len(ordered) != task_count or any(
        item.harness_role is not HarnessRole.FAULTY_PARENT for item in ordered
    ):
        raise ValueError("feedback requires the complete exact parent roster")
    linkage = feedback_linkage
    if linkage is FeedbackEvidenceLinkage.SCORE_ONLY:
        items: tuple[FaultFeedbackItem, ...] = ()
    elif linkage is FeedbackEvidenceLinkage.SOURCE_LINKED:
        items = tuple(_feedback_item(destination=item, source=item) for item in ordered)
    else:
        items = _shuffled_items(ordered)
    return HarnessFaultFeedbackProjection(
        plan_fingerprint=plan_fingerprint,
        linkage=linkage,
        parent_correct=sum(item.behavior_correct for item in ordered),
        task_count=task_count,
        items=items,
        authority_labeled_placebo=linkage is FeedbackEvidenceLinkage.SHUFFLED_PLACEBO,
    )


def _feedback_item(
    *,
    destination: FaultDevelopmentObservation,
    source: FaultDevelopmentObservation,
) -> FaultFeedbackItem:
    return FaultFeedbackItem(
        task_id=destination.task_id,
        source_task_id=source.task_id,
        family=destination.family,
        surface=destination.surface,
        scenario_role=source.scenario_role,
        raw_model_output_sha256=source.raw_model_output_sha256,
        parsed_output=source.parsed_output,
        inferred_branch=source.inferred_branch,
        behavior_correct=source.behavior_correct,
    )


def _shuffled_items(
    parent: tuple[FaultDevelopmentObservation, ...],
) -> tuple[FaultFeedbackItem, ...]:
    strata: dict[tuple[FaultFamily, str], list[FaultDevelopmentObservation]] = defaultdict(list)
    for item in parent:
        strata[(item.family, item.surface.value)].append(item)
    projected: list[FaultFeedbackItem] = []
    for key in sorted(strata, key=lambda item: (item[0].value, item[1])):
        members = sorted(strata[key], key=lambda item: item.task_id)
        if len(members) < 2:
            raise ValueError("shuffled placebo requires at least two items in every stratum")
        sources = members[1:] + members[:1]
        projected.extend(
            _feedback_item(destination=destination, source=source)
            for destination, source in zip(members, sources, strict=True)
        )
    return tuple(sorted(projected, key=lambda item: item.task_id))


def make_gate(
    *,
    plan: HarnessFaultDevelopmentPlan,
    proposal: ModelAuthoredFaultProposal,
    observations: tuple[FaultDevelopmentObservation, ...],
    parent_harness_ref: ArtifactRef,
    candidate_harness_ref: ArtifactRef,
    resource_budget_passed: bool,
) -> FaultDevelopmentGate:
    """Apply the cell's frozen promotion rule without a model or human ranker."""

    checked_plan = HarnessFaultDevelopmentPlan.model_validate(plan, strict=True)
    return make_gate_from_coordinates(
        promotion_rule=checked_plan.promotion_rule,
        task_count=checked_plan.task_count,
        proposal=proposal,
        observations=observations,
        parent_harness_ref=parent_harness_ref,
        candidate_harness_ref=candidate_harness_ref,
        resource_budget_passed=resource_budget_passed,
    )


def make_gate_from_coordinates(
    *,
    promotion_rule: PromotionRule,
    task_count: int,
    proposal: ModelAuthoredFaultProposal,
    observations: tuple[FaultDevelopmentObservation, ...],
    parent_harness_ref: ArtifactRef,
    candidate_harness_ref: ArtifactRef,
    resource_budget_passed: bool,
) -> FaultDevelopmentGate:
    """Apply a gate from exact evaluation coordinates without a synthetic call slot."""

    if type(promotion_rule) is not PromotionRule:
        raise TypeError("promotion_rule must be an exact PromotionRule")
    if type(task_count) is not int or task_count < 4:
        raise ValueError("task_count must be an integer of at least four")
    by_role = {
        role: tuple(item for item in observations if item.harness_role is role)
        for role in (
            HarnessRole.FAULTY_PARENT,
            HarnessRole.CANDIDATE,
            HarnessRole.REVERT,
            HarnessRole.PLACEBO,
        )
    }
    if any(len(items) != task_count for items in by_role.values()):
        raise ValueError("gate requires every attribution role over the complete roster")
    parent = by_role[HarnessRole.FAULTY_PARENT]
    candidate = by_role[HarnessRole.CANDIDATE]
    revert = by_role[HarnessRole.REVERT]
    placebo = by_role[HarnessRole.PLACEBO]
    protected_roles = {
        ScenarioRole.NULL_CONTROL,
        ScenarioRole.UNREPAIRABLE_CONTROL,
        ScenarioRole.DISTRACTOR_SHIFT_HARD_NEGATIVE,
    }
    parent_correct = sum(item.behavior_correct for item in parent)
    candidate_correct = sum(item.behavior_correct for item in candidate)
    revert_correct = sum(item.behavior_correct for item in revert)
    placebo_correct = sum(item.behavior_correct for item in placebo)
    protected_parent = sum(
        item.behavior_correct for item in parent if item.scenario_role in protected_roles
    )
    protected_candidate = sum(
        item.behavior_correct for item in candidate if item.scenario_role in protected_roles
    )
    performance_passed = bool(
        proposal.proposal_valid
        and resource_budget_passed
        and candidate_correct > parent_correct
        and protected_candidate >= protected_parent
    )
    repairable = tuple(
        item for item in candidate if item.scenario_role is ScenarioRole.REPAIRABLE_TARGET
    )
    nulls = tuple(item for item in candidate if item.scenario_role is ScenarioRole.NULL_CONTROL)
    shifts = tuple(
        item
        for item in candidate
        if item.scenario_role is ScenarioRole.DISTRACTOR_SHIFT_HARD_NEGATIVE
    )
    unrepairable = tuple(
        item for item in candidate if item.scenario_role is ScenarioRole.UNREPAIRABLE_CONTROL
    )
    activation_recall = _mean(item.inferred_branch is RuntimeBranch.SAFE for item in repairable)
    null_overactivation = _mean(item.inferred_branch is RuntimeBranch.SAFE for item in nulls)
    shift_accuracy = _mean(item.behavior_correct for item in shifts)
    unrepairable_correct = _mean(item.behavior_correct for item in unrepairable)
    parsed_rate = _mean(item.parsed_output is not None for item in candidate)
    mechanism_passed = bool(
        activation_recall == 1.0
        and null_overactivation == 0.0
        and shift_accuracy == 1.0
        and unrepairable_correct == 0.0
        and parsed_rate == 1.0
        and candidate_correct > revert_correct
        and candidate_correct > placebo_correct
    )
    promote = performance_passed and (
        promotion_rule is PromotionRule.PERFORMANCE_ONLY or mechanism_passed
    )
    return FaultDevelopmentGate(
        promotion_rule=promotion_rule,
        parent_correct=parent_correct,
        candidate_correct=candidate_correct,
        revert_correct=revert_correct,
        placebo_correct=placebo_correct,
        protected_parent_correct=protected_parent,
        protected_candidate_correct=protected_candidate,
        proposal_valid=proposal.proposal_valid,
        resource_budget_passed=resource_budget_passed,
        repair_activation_recall=activation_recall,
        null_overactivation_rate=null_overactivation,
        shift_route_accuracy=shift_accuracy,
        unrepairable_correct_rate=unrepairable_correct,
        parsed_output_rate=parsed_rate,
        performance_passed=performance_passed,
        mechanism_passed=mechanism_passed,
        decision=PromotionDecision.PROMOTE if promote else PromotionDecision.ROLLBACK,
        selected_harness_ref=candidate_harness_ref if promote else parent_harness_ref,
    )


def _mean(values) -> float:
    materialized = tuple(bool(value) for value in values)
    if not materialized:
        raise ValueError("mechanism rate requires a non-empty scenario role")
    return sum(materialized) / len(materialized)


__all__ = [
    "PROPOSER_PLANE_VERSION",
    "PROPOSER_SYSTEM_PROMPT",
    "PROPOSER_TASK_ID",
    "build_feedback_projection",
    "build_feedback_projection_from_coordinates",
    "make_gate",
    "make_gate_from_coordinates",
    "make_observation",
    "proposer_task",
    "proposer_visible_feedback",
    "publish_proposer_harness",
    "solver_seed",
]
