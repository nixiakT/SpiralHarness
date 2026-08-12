from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import spiral_harness.experiments.skill_probe_arms as skill_probe_arms
from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
)
from spiral_harness.core.lifecycle import CandidateState
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    BudgetPolicy,
    ComponentKind,
    HarnessComponentRef,
)
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
)
from spiral_harness.execution.materialization import HarnessMaterializer
from spiral_harness.execution.model import ReplayBackend, materialize_request, replay_key
from spiral_harness.execution.receipts import ExecutionReceiptIntegrityError
from spiral_harness.experiments.controller import (
    ExperimentBudgetError,
    ExperimentController,
    ExperimentControllerError,
)
from spiral_harness.experiments.controller_artifacts import (
    EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
    SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
    ExperimentUsageEntry,
    SkillProbeSettlementKind,
    SkillProbeUsageClaim,
    SkillProbeUsageSettlementClaim,
)
from spiral_harness.experiments.skill_probe_closure import (
    MatchedSkillProbeClosure,
    MatchedSkillProbeExecutionResult,
    SkillProbeShadowReport,
)
from spiral_harness.experiments.skill_probe_execution import (
    MATCHED_SKILL_PROBE_EXECUTION_ORDER_FINGERPRINT,
    MATCHED_SKILL_PROBE_RESET_FINGERPRINT,
    MatchedSkillProbeExecution,
    MatchedSkillProbeExecutionError,
)
from spiral_harness.harness.registry import HarnessRegistry
from spiral_harness.skills.package import SKILL_PACKAGE_MEDIA_TYPE
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.storage.journal import CandidateJournal
from spiral_harness.verification.mechanism import ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE
from spiral_harness.verification.models import MechanismEvidence
from spiral_harness.verification.skill_plan import (
    SKILL_MECHANISM_PLAN_MEDIA_TYPE,
    SkillMechanismClaim,
    SkillPlaceboControl,
)
from tests.unit.test_skill_probe_preregistration import (
    ProbeFixture,
    _controller_at_probes,
    _probe_fixture,
    _put,
)

_BACKEND = "matched-skill-probe-replay@sha256:fixed-v1"


@dataclass(frozen=True)
class ExecutableProbe:
    graph: ProbeFixture
    spec: FrozenModelSpec


def _spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="deterministic-replay",
        backend_fingerprint=_BACKEND,
        model="hosted/matched-skill-probe-model",
        revision="snapshot-2026-08-12",
        tokenizer="provider/matched-skill-probe-tokenizer",
        tokenizer_revision="snapshot-2026-08-12",
        runtime="fixed-runtime-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=32,
            timeout_seconds=5.0,
        ),
    )


def _replace[ModelT: BaseModel](model: ModelT, **updates: object) -> ModelT:
    values = model.model_dump(mode="python", round_trip=True, warnings="none")
    values.update(updates)
    return type(model).model_validate(values, strict=True)


def _executable_probe(root: Path) -> ExecutableProbe:
    base = _probe_fixture(root)
    store = base.store
    spec = _spec()
    before_package = _replace(
        base.before_package,
        compatible_model_fingerprints=(spec.model_fingerprint,),
        runtime_fingerprints=(spec.runtime_fingerprint,),
    )
    before_ref = _put(store, before_package, SKILL_PACKAGE_MEDIA_TYPE)
    after_package = _replace(
        base.after_package,
        parent_package_ref=before_ref,
        compatible_model_fingerprints=(spec.model_fingerprint,),
        runtime_fingerprints=(spec.runtime_fingerprint,),
    )
    after_ref = _put(store, after_package, SKILL_PACKAGE_MEDIA_TYPE)
    before_component = HarnessComponentRef(
        name=base.mutation.before.name,
        kind=base.mutation.before.kind,
        artifact=before_ref,
    )
    after_component = HarnessComponentRef(
        name=base.mutation.after.name,
        kind=base.mutation.after.kind,
        artifact=after_ref,
    )
    prompt_ref = store.put_bytes(
        b"Solve the task and return only the final answer.",
        media_type="text/plain",
    )
    prompt_component = HarnessComponentRef(
        name="system-prompt",
        kind=ComponentKind.PROMPT,
        artifact=prompt_ref,
    )
    parent = _replace(
        base.parent,
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        components=(prompt_component, before_component),
    )
    parent_ref = _put(store, parent, HARNESS_MANIFEST_MEDIA_TYPE)
    mutation = _replace(
        base.mutation,
        before=before_component,
        after=after_component,
    )
    mutation_ref = _put(store, mutation, base.mutation_ref.media_type)
    child = HarnessRegistry(base.experiment.mutation_policy).apply_mutation(
        parent=parent,
        parent_ref=parent_ref,
        mutation=mutation,
        artifact_bytes=store.get_bytes(after_ref),
        artifact_media_type=SKILL_PACKAGE_MEDIA_TYPE,
    )
    child_ref = _put(store, child, HARNESS_MANIFEST_MEDIA_TYPE)
    placebo_package = _replace(
        base.placebo_package,
        parent_package_ref=before_ref,
        compatible_model_fingerprints=(spec.model_fingerprint,),
        runtime_fingerprints=(spec.runtime_fingerprint,),
    )
    placebo_package_ref = _put(store, placebo_package, SKILL_PACKAGE_MEDIA_TYPE)
    placebo_component = HarnessComponentRef(
        name=after_component.name,
        kind=after_component.kind,
        artifact=placebo_package_ref,
    )
    placebo_harness = _replace(
        child,
        components=(prompt_component, placebo_component),
    )
    placebo_harness_ref = _put(store, placebo_harness, HARNESS_MANIFEST_MEDIA_TYPE)

    materializer = HarnessMaterializer(store, spec=spec)
    candidate_disclosure = materializer.materialize(child_ref).skill_disclosure
    placebo_disclosure = materializer.materialize(placebo_harness_ref).skill_disclosure
    assert candidate_disclosure is not None and placebo_disclosure is not None
    placebo_control = SkillPlaceboControl(
        evidence_profile=base.policy.evidence_profile,
        candidate_package_ref=after_ref,
        placebo_package_ref=placebo_package_ref,
        candidate_harness_ref=child_ref,
        placebo_harness_ref=placebo_harness_ref,
        neutral_rules_ref=base.neutral_ref,
        changed_rule_ids=base.placebo_control.changed_rule_ids,
        candidate_rule_count=len(after_package.rules),
        placebo_rule_count=len(placebo_package.rules),
        candidate_context_size_bytes=candidate_disclosure.context_size_bytes,
        placebo_context_size_bytes=placebo_disclosure.context_size_bytes,
    )
    placebo_control_ref = _put(store, placebo_control, base.placebo_control_ref.media_type)
    policy = _replace(
        base.policy,
        model_fingerprint=spec.model_fingerprint,
        model_spec_fingerprint=spec.fingerprint,
        inference_fingerprint=spec.inference_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        reset_fingerprint=MATCHED_SKILL_PROBE_RESET_FINGERPRINT,
        execution_order_fingerprint=MATCHED_SKILL_PROBE_EXECUTION_ORDER_FINGERPRINT,
    )
    policy_ref = _put(store, policy, base.policy_ref.media_type)
    protocol = _replace(
        base.protocol,
        model_fingerprint=spec.model_fingerprint,
        model_spec_fingerprint=spec.fingerprint,
        inference_fingerprint=spec.inference_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        skill_verification_policy_ref=policy_ref,
    )
    protocol_ref = _put(store, protocol, PROTOCOL_MANIFEST_MEDIA_TYPE)
    experiment = _replace(
        base.experiment,
        protocol_ref=protocol_ref,
        seed_harness_ref=parent_ref,
    )
    experiment_ref = _put(store, experiment, EXPERIMENT_MANIFEST_MEDIA_TYPE)
    candidate = _replace(
        base.candidate,
        experiment_ref=experiment_ref,
        parent_harness_ref=parent_ref,
        child_harness_ref=child_ref,
        mutation_ref=mutation_ref,
    )
    candidate_ref = _put(store, candidate, CANDIDATE_MANIFEST_MEDIA_TYPE)
    revert_schedule = _replace(
        base.plan.revert_schedule,
        parent_harness_id=parent_ref.sha256,
        candidate_harness_id=child_ref.sha256,
    )
    placebo_schedule = _replace(
        base.plan.placebo_schedule,
        parent_harness_id=placebo_harness_ref.sha256,
        candidate_harness_id=child_ref.sha256,
    )
    plan = _replace(
        base.plan,
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        candidate_ref=candidate_ref,
        mutation_ref=mutation_ref,
        parent_harness_ref=parent_ref,
        candidate_harness_ref=child_ref,
        before_skill_package_ref=before_ref,
        after_skill_package_ref=after_ref,
        policy_ref=policy_ref,
        policy_fingerprint=policy.fingerprint,
        placebo_control_ref=placebo_control_ref,
        placebo_harness_ref=placebo_harness_ref,
        model_spec_fingerprint=spec.fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        reset_fingerprint=MATCHED_SKILL_PROBE_RESET_FINGERPRINT,
        execution_order_fingerprint=MATCHED_SKILL_PROBE_EXECUTION_ORDER_FINGERPRINT,
        revert_schedule=revert_schedule,
        placebo_schedule=placebo_schedule,
    )
    plan_ref = _put(store, plan, SKILL_MECHANISM_PLAN_MEDIA_TYPE)
    graph = _replace_fixture(
        base,
        policy=policy,
        policy_ref=policy_ref,
        protocol=protocol,
        protocol_ref=protocol_ref,
        experiment=experiment,
        experiment_ref=experiment_ref,
        before_package=before_package,
        before_ref=before_ref,
        after_package=after_package,
        after_ref=after_ref,
        parent=parent,
        parent_ref=parent_ref,
        mutation=mutation,
        mutation_ref=mutation_ref,
        child=child,
        child_ref=child_ref,
        placebo_package=placebo_package,
        placebo_package_ref=placebo_package_ref,
        placebo_harness=placebo_harness,
        placebo_harness_ref=placebo_harness_ref,
        placebo_control=placebo_control,
        placebo_control_ref=placebo_control_ref,
        candidate=candidate,
        candidate_ref=candidate_ref,
        plan=plan,
        plan_ref=plan_ref,
    )
    return ExecutableProbe(graph=graph, spec=spec)


def _replace_fixture(base: ProbeFixture, **updates: object) -> ProbeFixture:
    values = {field: getattr(base, field) for field in base.__dataclass_fields__}
    values.update(updates)
    return ProbeFixture(**values)


def _backend(graph: ProbeFixture, spec: FrozenModelSpec, schedule_name: str) -> ReplayBackend:
    plan = graph.plan
    schedule = plan.revert_schedule if schedule_name == "revert" else plan.placebo_schedule
    parent_ref = plan.parent_harness_ref if schedule_name == "revert" else plan.placebo_harness_ref
    materializer = HarnessMaterializer(graph.store, spec=spec)
    harnesses = {
        side: materializer.materialize(
            parent_ref if side.value == "parent" else plan.candidate_harness_ref
        )
        for side in schedule.sides
    }
    responses = {}
    for cell in schedule.iter_cells():
        request = materialize_request(
            graph.task.candidate_task,
            harnesses[cell.side],
            seed=schedule.seed_for(cell, attempt_index=0),
        )
        responses[replay_key(spec, request)] = BackendResponse(
            output=f"{schedule_name}:{cell.side.value}:42",
            usage=BackendTokenUsage(input_tokens=2, output_tokens=1),
        )
    return ReplayBackend(fingerprint=_BACKEND, responses=responses)


def _run(tmp_path: Path) -> tuple[ExecutableProbe, object, MatchedSkillProbeExecution]:
    executable = _executable_probe(tmp_path)
    graph = executable.graph
    controller, _, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    execution = controller.execute_matched_skill_probes(
        authorization_ref=authorization_ref,
        model_spec=executable.spec,
        revert_backend=_backend(graph, executable.spec, "revert"),
        placebo_backend=_backend(graph, executable.spec, "placebo"),
    )
    return executable, controller, execution


def test_executes_both_preregistered_schedules_and_closes_request_inclusion(
    tmp_path: Path,
) -> None:
    executable, controller, execution = _run(tmp_path)
    graph = executable.graph
    verified = controller.verify_matched_skill_probe_result(execution)
    closure = verified.closure
    shadow = verified.shadow_report

    assert closure.promotion_authority is False
    assert closure.revert.usage.settled_attempts == graph.plan.revert_schedule.cell_count
    assert closure.placebo.usage.settled_attempts == graph.plan.placebo_schedule.cell_count
    assert closure.revert.ledger_id != closure.placebo.ledger_id
    assert closure.revert.writer_epoch_id != closure.placebo.writer_epoch_id
    assert closure.revert.receipt_refs != closure.placebo.receipt_refs
    assert shadow.reported_claims == (SkillMechanismClaim.REQUEST_INCLUSION,)
    assert len(shadow.unavailable_claims) == 6
    assert all(
        call_count == graph.plan.revert_schedule.cell_count
        for call_count in (
            closure.revert.usage.attempt_count,
            closure.placebo.usage.attempt_count,
        )
    )


def test_closure_rejects_swapped_same_and_reopened_ledgers(tmp_path: Path) -> None:
    executable, controller, execution = _run(tmp_path)
    graph = executable.graph

    with pytest.raises(ExperimentControllerError, match="matched arm replay failed"):
        controller.verify_matched_skill_probe_result(
            MatchedSkillProbeExecution(
                result=execution.result,
                revert_attempt_ledger=execution.placebo_attempt_ledger,
                placebo_attempt_ledger=execution.revert_attempt_ledger,
            )
        )

    with pytest.raises(
        ExperimentControllerError,
        match="matched arms reuse one live ledger",
    ):
        controller.verify_matched_skill_probe_result(
            MatchedSkillProbeExecution(
                result=execution.result,
                revert_attempt_ledger=execution.revert_attempt_ledger,
                placebo_attempt_ledger=execution.revert_attempt_ledger,
            )
        )

    revert_ledger = execution.revert_attempt_ledger
    reopened_revert = AttemptLedger(
        graph.store,
        ledger_id=revert_ledger.ledger_id,
        budget=revert_ledger.budget,
        tail_ref=revert_ledger.tail_ref,
    )
    with pytest.raises(
        ExperimentControllerError,
        match="matched arm replay failed",
    ) as reopened_error:
        controller.verify_matched_skill_probe_result(
            MatchedSkillProbeExecution(
                result=execution.result,
                revert_attempt_ledger=reopened_revert,
                placebo_attempt_ledger=execution.placebo_attempt_ledger,
            )
        )
    replay_error = reopened_error.value.__cause__
    assert isinstance(replay_error, MatchedSkillProbeExecutionError)
    assert isinstance(replay_error.__cause__, ExecutionReceiptIntegrityError)
    assert "writer epoch" in str(replay_error.__cause__)


def test_closure_rejects_swapped_arms_preflights_and_inclusion_evidence(
    tmp_path: Path,
) -> None:
    executable, controller, execution = _run(tmp_path)
    graph = executable.graph
    closure = graph.store.get_json(execution.result.closure_ref, MatchedSkillProbeClosure)

    swapped_arm_values = closure.model_dump(
        mode="python",
        round_trip=True,
        warnings="none",
    )
    swapped_arm_values["revert"], swapped_arm_values["placebo"] = (
        swapped_arm_values["placebo"],
        swapped_arm_values["revert"],
    )
    with pytest.raises(ValidationError, match="closure arms are mislabeled"):
        MatchedSkillProbeClosure.model_validate(swapped_arm_values, strict=True)

    relabeled = _replace(
        closure,
        revert=_replace(closure.placebo, control="revert"),
        placebo=_replace(closure.revert, control="placebo"),
    )
    relabeled_ref = graph.store.put_json(
        relabeled,
        media_type=execution.result.closure_ref.media_type,
    )
    with pytest.raises(ExperimentControllerError, match="binds another closure"):
        controller.verify_matched_skill_probe_result(
            MatchedSkillProbeExecution(
                result=MatchedSkillProbeExecutionResult(
                    closure_ref=relabeled_ref,
                    shadow_report_ref=execution.result.shadow_report_ref,
                ),
                revert_attempt_ledger=execution.placebo_attempt_ledger,
                placebo_attempt_ledger=execution.revert_attempt_ledger,
            )
        )

    for field_name in ("preflight_ref", "request_inclusion_ref"):
        forged = _replace(
            closure,
            revert=_replace(
                closure.revert,
                **{field_name: getattr(closure.placebo, field_name)},
            ),
            placebo=_replace(
                closure.placebo,
                **{field_name: getattr(closure.revert, field_name)},
            ),
        )
        forged_ref = graph.store.put_json(
            forged,
            media_type=execution.result.closure_ref.media_type,
        )
        with pytest.raises(ExperimentControllerError, match="binds another closure"):
            controller.verify_matched_skill_probe_result(
                MatchedSkillProbeExecution(
                    result=MatchedSkillProbeExecutionResult(
                        closure_ref=forged_ref,
                        shadow_report_ref=execution.result.shadow_report_ref,
                    ),
                    revert_attempt_ledger=execution.revert_attempt_ledger,
                    placebo_attempt_ledger=execution.placebo_attempt_ledger,
                )
            )


def test_forged_scope_cannot_claim_activation_behavior_or_promotion(tmp_path: Path) -> None:
    executable, controller, execution = _run(tmp_path)
    store = executable.graph.store
    closure = store.get_json(execution.result.closure_ref, MatchedSkillProbeClosure)
    values = closure.model_dump(mode="python", round_trip=True, warnings="none")
    values["promotion_authority"] = True
    forged_closure_ref = store.put_json(
        values,
        media_type=execution.result.closure_ref.media_type,
    )
    with pytest.raises(
        ExperimentControllerError,
        match="matched skill probe closure cannot be loaded exactly",
    ):
        controller.verify_matched_skill_probe_result(
            MatchedSkillProbeExecution(
                result=MatchedSkillProbeExecutionResult(
                    closure_ref=forged_closure_ref,
                    shadow_report_ref=execution.result.shadow_report_ref,
                ),
                revert_attempt_ledger=execution.revert_attempt_ledger,
                placebo_attempt_ledger=execution.placebo_attempt_ledger,
            )
        )

    shadow = store.get_json(execution.result.shadow_report_ref, SkillProbeShadowReport)
    forged_shadow_updates = (
        {"promotion_authority": True},
        {"reported_claims": (SkillMechanismClaim.RUNTIME_ACTIVATION,)},
        {"reported_claims": (SkillMechanismClaim.BEHAVIOR,)},
    )
    for updates in forged_shadow_updates:
        shadow_values = shadow.model_dump(mode="python", round_trip=True, warnings="none")
        shadow_values.update(updates)
        forged_shadow_ref = store.put_json(
            shadow_values,
            media_type=execution.result.shadow_report_ref.media_type,
        )
        with pytest.raises(
            ExperimentControllerError,
            match="skill probe shadow report cannot be loaded exactly",
        ):
            controller.verify_matched_skill_probe_result(
                MatchedSkillProbeExecution(
                    result=MatchedSkillProbeExecutionResult(
                        closure_ref=execution.result.closure_ref,
                        shadow_report_ref=forged_shadow_ref,
                    ),
                    revert_attempt_ledger=execution.revert_attempt_ledger,
                    placebo_attempt_ledger=execution.placebo_attempt_ledger,
                )
            )


def test_foreign_repository_object_and_second_execution_fail_closed(tmp_path: Path) -> None:
    executable = _executable_probe(tmp_path)
    graph = executable.graph
    controller, _, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    revert_backend = _backend(graph, executable.spec, "revert")
    placebo_backend = _backend(graph, executable.spec, "placebo")
    kwargs = {
        "authorization_ref": authorization_ref,
        "model_spec": executable.spec,
        "revert_backend": revert_backend,
        "placebo_backend": placebo_backend,
    }
    foreign_repository = ArtifactStore(graph.store.root)

    execution = controller.execute_matched_skill_probes(**kwargs)
    foreign_revert = AttemptLedger(
        foreign_repository,
        ledger_id=execution.revert_attempt_ledger.ledger_id,
        budget=execution.revert_attempt_ledger.budget,
        tail_ref=execution.revert_attempt_ledger.tail_ref,
    )
    with pytest.raises(ExperimentControllerError, match="another repository object"):
        controller.verify_matched_skill_probe_result(
            MatchedSkillProbeExecution(
                result=execution.result,
                revert_attempt_ledger=foreign_revert,
                placebo_attempt_ledger=execution.placebo_attempt_ledger,
            )
        )

    calls_before_replay = (len(revert_backend.calls), len(placebo_backend.calls))
    usage_before_replay = controller.current_usage()
    with pytest.raises(MatchedSkillProbeExecutionError, match="already started"):
        controller.execute_matched_skill_probes(**kwargs)
    assert controller.current_usage() == usage_before_replay
    assert (len(revert_backend.calls), len(placebo_backend.calls)) == calls_before_replay


def test_shared_backend_session_is_rejected_before_authorization_consumption(
    tmp_path: Path,
) -> None:
    executable = _executable_probe(tmp_path)
    graph = executable.graph
    controller, _, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    shared_backend = _backend(graph, executable.spec, "revert")

    with pytest.raises(MatchedSkillProbeExecutionError, match="independent backend session"):
        controller.execute_matched_skill_probes(
            authorization_ref=authorization_ref,
            model_spec=executable.spec,
            revert_backend=shared_backend,
            placebo_backend=shared_backend,
        )

    assert not shared_backend.calls
    assert controller.usage_tail_ref is None
    execution = controller.execute_matched_skill_probes(
        authorization_ref=authorization_ref,
        model_spec=executable.spec,
        revert_backend=_backend(graph, executable.spec, "revert"),
        placebo_backend=_backend(graph, executable.spec, "placebo"),
    )
    controller.verify_matched_skill_probe_result(execution)


def test_second_arm_preflight_failure_precedes_backends_budget_and_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable_probe(tmp_path)
    graph = executable.graph
    controller, _, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    revert_backend = _backend(graph, executable.spec, "revert")
    placebo_backend = _backend(graph, executable.spec, "placebo")
    publish = skill_probe_arms.publish_schedule_preflight
    prepared = 0

    def fail_second_preflight(*args: object, **kwargs: object) -> ArtifactRef:
        nonlocal prepared
        prepared += 1
        if prepared == 2:
            raise RuntimeError("injected second-arm preflight failure")
        return publish(*args, **kwargs)

    monkeypatch.setattr(skill_probe_arms, "publish_schedule_preflight", fail_second_preflight)
    kwargs = {
        "authorization_ref": authorization_ref,
        "model_spec": executable.spec,
        "revert_backend": revert_backend,
        "placebo_backend": placebo_backend,
    }
    with pytest.raises(MatchedSkillProbeExecutionError, match="placebo arm preparation failed"):
        controller.execute_matched_skill_probes(**kwargs)

    assert prepared == 2
    assert not revert_backend.calls
    assert not placebo_backend.calls
    assert controller.usage_tail_ref is None
    assert controller.current_usage().entry_count == 0

    monkeypatch.undo()
    execution = controller.execute_matched_skill_probes(**kwargs)
    controller.verify_matched_skill_probe_result(execution)


def _budgeted_executable(root: Path, **budget_limits: object) -> ExecutableProbe:
    executable = _executable_probe(root)
    graph = executable.graph
    budget = BudgetPolicy.model_validate(budget_limits, strict=True)
    protocol = _replace(graph.protocol, budget=budget)
    protocol_ref = _put(graph.store, protocol, PROTOCOL_MANIFEST_MEDIA_TYPE)
    experiment = _replace(
        graph.experiment,
        protocol_ref=protocol_ref,
        search_budget=budget,
    )
    experiment_ref = _put(graph.store, experiment, EXPERIMENT_MANIFEST_MEDIA_TYPE)
    candidate = _replace(graph.candidate, experiment_ref=experiment_ref)
    candidate_ref = _put(graph.store, candidate, CANDIDATE_MANIFEST_MEDIA_TYPE)
    plan = _replace(
        graph.plan,
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        candidate_ref=candidate_ref,
    )
    plan_ref = _put(graph.store, plan, SKILL_MECHANISM_PLAN_MEDIA_TYPE)
    return ExecutableProbe(
        graph=_replace_fixture(
            graph,
            protocol=protocol,
            protocol_ref=protocol_ref,
            experiment=experiment,
            experiment_ref=experiment_ref,
            candidate=candidate,
            candidate_ref=candidate_ref,
            plan=plan,
            plan_ref=plan_ref,
        ),
        spec=executable.spec,
    )


def _probe_reservation(graph: ProbeFixture) -> tuple[int, int]:
    schedules = (graph.plan.revert_schedule, graph.plan.placebo_schedule)
    return (
        sum(schedule.required_attempts for schedule in schedules),
        sum(schedule.required_tokens for schedule in schedules),
    )


def _authorized_execution(
    executable: ExecutableProbe,
) -> tuple[ExperimentController, ArtifactRef, ReplayBackend, ReplayBackend]:
    graph = executable.graph
    controller, _, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    return (
        controller,
        authorization_ref,
        _backend(graph, executable.spec, "revert"),
        _backend(graph, executable.spec, "placebo"),
    )


def test_probe_reserves_both_arms_in_experiment_usage_without_counting_a_gate_query(
    tmp_path: Path,
) -> None:
    executable = _budgeted_executable(
        tmp_path,
        max_evaluations=10,
        max_tokens=2_051,
    )
    graph = executable.graph
    expected_evaluations, expected_tokens = _probe_reservation(graph)
    controller, authorization_ref, revert_backend, placebo_backend = _authorized_execution(
        executable
    )

    execution = controller.execute_matched_skill_probes(
        authorization_ref=authorization_ref,
        model_spec=executable.spec,
        revert_backend=revert_backend,
        placebo_backend=placebo_backend,
    )
    usage = controller.current_usage()
    claim_ref = usage.skill_probe_claim_refs[0]
    claim = graph.store.get_json(claim_ref, SkillProbeUsageClaim)
    closure = graph.store.get_json(execution.result.closure_ref, MatchedSkillProbeClosure)
    actual_tokens = closure.revert.usage.charged_tokens + closure.placebo.usage.charged_tokens

    assert (expected_evaluations, expected_tokens) == (8, 2_048)
    assert usage.entry_count == 2
    assert usage.query_count == 0
    assert usage.claim_refs == ()
    assert usage.candidate_refs == ()
    assert usage.evaluation_refs == ()
    assert usage.skill_probe_claim_refs == (claim_ref,)
    assert usage.skill_probe_authorization_refs == (authorization_ref,)
    assert len(usage.skill_probe_settlement_refs) == 1
    settlement = graph.store.get_json(
        usage.skill_probe_settlement_refs[0],
        SkillProbeUsageSettlementClaim,
    )
    assert settlement.terminal_kind is SkillProbeSettlementKind.COMPLETED
    assert settlement.closure_ref == execution.result.closure_ref
    assert usage.total_evaluations == claim.evaluation_units == expected_evaluations
    assert usage.total_tokens == claim.tokens == expected_tokens
    assert usage.total_tokens > actual_tokens
    assert usage.remaining_evaluations == 2
    assert usage.remaining_tokens == 3
    assert claim.plan_ref == graph.plan_ref
    assert claim.revert_schedule_fingerprint == graph.plan.revert_schedule.fingerprint
    assert claim.placebo_schedule_fingerprint == graph.plan.placebo_schedule.fingerprint
    assert controller.query_usage(controller.usage_tail_ref) == usage


@pytest.mark.parametrize(
    ("budget_limits", "error"),
    (
        ({"max_evaluations": 7}, "max_evaluations"),
        (
            {"max_evaluations": 8, "max_tokens": 2_047},
            "max_tokens",
        ),
    ),
)
def test_probe_budget_shortfall_rejects_before_either_backend_or_grant_is_consumed(
    tmp_path: Path,
    budget_limits: dict[str, object],
    error: str,
) -> None:
    executable = _budgeted_executable(tmp_path, **budget_limits)
    controller, authorization_ref, revert_backend, placebo_backend = _authorized_execution(
        executable
    )

    with pytest.raises(ExperimentBudgetError, match=error):
        controller.execute_matched_skill_probes(
            authorization_ref=authorization_ref,
            model_spec=executable.spec,
            revert_backend=revert_backend,
            placebo_backend=placebo_backend,
        )

    assert not revert_backend.calls
    assert not placebo_backend.calls
    assert controller.usage_tail_ref is None
    assert controller.current_usage().entry_count == 0

    # Repeating the exact request proves budget rejection did not consume the
    # one-shot execution grant. Frozen experiment ceilings are not mutable.
    with pytest.raises(ExperimentBudgetError, match=error):
        controller.execute_matched_skill_probes(
            authorization_ref=authorization_ref,
            model_spec=executable.spec,
            revert_backend=revert_backend,
            placebo_backend=placebo_backend,
        )
    assert not revert_backend.calls
    assert not placebo_backend.calls


@pytest.mark.parametrize(
    ("bounded_resource", "error"),
    (
        ("max_wall_time_seconds", "wall-time ceiling"),
        ("max_cost_usd", "cost ceiling"),
    ),
)
def test_probe_without_preregistered_wall_or_cost_bound_fails_before_backend_calls(
    tmp_path: Path,
    bounded_resource: str,
    error: str,
) -> None:
    executable = _budgeted_executable(
        tmp_path,
        max_evaluations=8,
        **{bounded_resource: 1.0},
    )
    controller, authorization_ref, revert_backend, placebo_backend = _authorized_execution(
        executable
    )

    with pytest.raises(ExperimentBudgetError, match=error):
        controller.execute_matched_skill_probes(
            authorization_ref=authorization_ref,
            model_spec=executable.spec,
            revert_backend=revert_backend,
            placebo_backend=placebo_backend,
        )

    assert not revert_backend.calls
    assert not placebo_backend.calls
    assert controller.usage_tail_ref is None


def test_failed_second_arm_keeps_reservation_and_retry_does_not_charge_or_call_again(
    tmp_path: Path,
) -> None:
    executable = _budgeted_executable(
        tmp_path,
        max_evaluations=8,
        max_tokens=2_048,
    )
    graph = executable.graph
    expected_evaluations, expected_tokens = _probe_reservation(graph)
    controller, authorization_ref, revert_backend, _ = _authorized_execution(executable)
    placebo_backend = ReplayBackend(fingerprint=_BACKEND)
    kwargs = {
        "authorization_ref": authorization_ref,
        "model_spec": executable.spec,
        "revert_backend": revert_backend,
        "placebo_backend": placebo_backend,
    }

    with pytest.raises(MatchedSkillProbeExecutionError, match="did not settle"):
        controller.execute_matched_skill_probes(**kwargs)

    charged = controller.current_usage()
    calls_after_failure = (len(revert_backend.calls), len(placebo_backend.calls))
    assert calls_after_failure == (graph.plan.revert_schedule.cell_count, 2)
    assert charged.entry_count == 2
    assert charged.query_count == 0
    assert charged.total_evaluations == expected_evaluations
    assert charged.total_tokens == expected_tokens
    assert charged.remaining_evaluations == 0
    assert charged.remaining_tokens == 0
    assert charged.skill_probe_authorization_refs == (authorization_ref,)
    assert len(charged.skill_probe_settlement_refs) == 1
    settlement = graph.store.get_json(
        charged.skill_probe_settlement_refs[0],
        SkillProbeUsageSettlementClaim,
    )
    assert settlement.terminal_kind is SkillProbeSettlementKind.FAILED
    assert settlement.closure_ref is None
    assert settlement.revert.terminal_tail_ref is not None
    assert settlement.placebo.terminal_tail_ref is not None

    with pytest.raises(MatchedSkillProbeExecutionError, match="already started"):
        controller.execute_matched_skill_probes(**kwargs)

    assert controller.current_usage() == charged
    assert (len(revert_backend.calls), len(placebo_backend.calls)) == calls_after_failure


def test_first_arm_token_overrun_stops_siblings_and_poisons_global_usage(
    tmp_path: Path,
) -> None:
    executable = _budgeted_executable(
        tmp_path,
        max_evaluations=16,
        max_tokens=3_000,
    )
    graph = executable.graph
    _, reserved_tokens = _probe_reservation(graph)
    overrun_tokens = reserved_tokens + 13
    controller, authorization_ref, _, placebo_backend = _authorized_execution(executable)
    revert_backend = ReplayBackend(
        fingerprint=_BACKEND,
        default_response=BackendResponse(
            output="overrun",
            usage=BackendTokenUsage(input_tokens=overrun_tokens - 1, output_tokens=1),
        ),
    )
    kwargs = {
        "authorization_ref": authorization_ref,
        "model_spec": executable.spec,
        "revert_backend": revert_backend,
        "placebo_backend": placebo_backend,
    }

    with pytest.raises(MatchedSkillProbeExecutionError, match="token ceiling"):
        controller.execute_matched_skill_probes(**kwargs)

    charged = controller.current_usage()
    calls_after_overrun = (len(revert_backend.calls), len(placebo_backend.calls))
    assert calls_after_overrun == (1, 0)
    assert charged.entry_count == 2
    assert charged.total_evaluations == graph.plan.revert_schedule.required_attempts + (
        graph.plan.placebo_schedule.required_attempts
    )
    assert charged.total_tokens == overrun_tokens
    assert charged.poisoned is True
    assert charged.accounting_complete is True
    assert charged.remaining_tokens == 0
    assert charged.skill_probe_authorization_refs == (authorization_ref,)
    assert len(charged.skill_probe_settlement_refs) == 1
    settlement = graph.store.get_json(
        charged.skill_probe_settlement_refs[0],
        SkillProbeUsageSettlementClaim,
    )
    assert settlement.terminal_kind is SkillProbeSettlementKind.FAILED
    assert settlement.closure_ref is None
    assert settlement.reserved_tokens == reserved_tokens
    assert settlement.encumbered_tokens == overrun_tokens
    assert settlement.token_adjustment == overrun_tokens - reserved_tokens
    assert settlement.poisoned is True
    assert settlement.revert.encumbered_tokens == overrun_tokens
    assert settlement.revert.poisoned is True
    assert settlement.revert.terminal_tail_ref is not None
    assert settlement.placebo.encumbered_tokens == 0
    assert settlement.placebo.poisoned is False
    assert settlement.placebo.terminal_tail_ref is None

    with pytest.raises(ExperimentControllerError, match="usage accounting is blocked"):
        controller.execute_matched_skill_probes(**kwargs)
    assert (len(revert_backend.calls), len(placebo_backend.calls)) == calls_after_overrun

    assert charged.tail_ref is not None
    restarted = ExperimentController(
        graph.store,
        experiment_ref=graph.experiment_ref,
        gate_batch_verifier=graph.gate_batch_service.verification_capability,
        mechanism_evidence_verifier=graph.mechanism_service.verification_capability,
        usage_tail_ref=charged.tail_ref,
    )
    fresh_revert = _backend(graph, executable.spec, "revert")
    fresh_placebo = _backend(graph, executable.spec, "placebo")

    assert restarted.current_usage() == charged
    with pytest.raises(ExperimentControllerError, match="usage accounting is blocked"):
        restarted.execute_matched_skill_probes(
            authorization_ref=authorization_ref,
            model_spec=executable.spec,
            revert_backend=fresh_revert,
            placebo_backend=fresh_placebo,
        )
    assert not fresh_revert.calls
    assert not fresh_placebo.calls


def test_settlement_failure_during_execution_failure_is_primary_and_single_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _budgeted_executable(
        tmp_path,
        max_evaluations=16,
        max_tokens=3_000,
    )
    graph = executable.graph
    _, reserved_tokens = _probe_reservation(graph)
    overrun_tokens = reserved_tokens + 13
    controller, authorization_ref, _, placebo_backend = _authorized_execution(executable)
    revert_backend = ReplayBackend(
        fingerprint=_BACKEND,
        default_response=BackendResponse(
            output="overrun",
            usage=BackendTokenUsage(input_tokens=overrun_tokens - 1, output_tokens=1),
        ),
    )
    settlement_calls = 0

    def fail_settlement(**kwargs: object) -> ArtifactRef:
        nonlocal settlement_calls
        settlement_calls += 1
        assert kwargs["terminal_kind"] is SkillProbeSettlementKind.FAILED
        assert kwargs["revert_terminal_tail_ref"] is not None
        assert kwargs["placebo_terminal_tail_ref"] is None
        raise RuntimeError("settlement store unavailable")

    monkeypatch.setattr(controller._usage_ledger, "settle_skill_probe", fail_settlement)
    kwargs = {
        "authorization_ref": authorization_ref,
        "model_spec": executable.spec,
        "revert_backend": revert_backend,
        "placebo_backend": placebo_backend,
    }

    with pytest.raises(
        ExperimentControllerError,
        match="skill-probe usage settlement publication failed: settlement store unavailable",
    ) as error:
        controller.execute_matched_skill_probes(**kwargs)

    assert isinstance(error.value.__cause__, MatchedSkillProbeExecutionError)
    assert "token ceiling" in str(error.value.__cause__)
    assert settlement_calls == 1
    assert (len(revert_backend.calls), len(placebo_backend.calls)) == (1, 0)
    charged = controller.current_usage()
    assert charged.entry_count == 1
    assert charged.poisoned is False
    assert charged.total_tokens == reserved_tokens

    with pytest.raises(ExperimentControllerError, match="usage accounting is blocked"):
        controller.execute_matched_skill_probes(**kwargs)
    assert settlement_calls == 1
    assert (len(revert_backend.calls), len(placebo_backend.calls)) == (1, 0)


def test_probe_usage_replay_rejects_a_tampered_schedule_reservation(
    tmp_path: Path,
) -> None:
    executable = _budgeted_executable(
        tmp_path,
        max_evaluations=8,
        max_tokens=2_048,
    )
    graph = executable.graph
    controller, authorization_ref, revert_backend, placebo_backend = _authorized_execution(
        executable
    )
    controller.execute_matched_skill_probes(
        authorization_ref=authorization_ref,
        model_spec=executable.spec,
        revert_backend=revert_backend,
        placebo_backend=placebo_backend,
    )
    usage = controller.current_usage()
    settlement_entry = graph.store.get_json(usage.tail_ref, ExperimentUsageEntry)
    reservation_entry = graph.store.get_json(
        settlement_entry.previous_entry_ref,
        ExperimentUsageEntry,
    )
    claim = graph.store.get_json(usage.skill_probe_claim_refs[0], SkillProbeUsageClaim)
    tampered_claim = _replace(claim, revert_schedule_fingerprint="0" * 64)
    tampered_claim_ref = graph.store.put_json(
        tampered_claim,
        media_type=SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
    )
    tampered_reservation_tail = graph.store.put_json(
        _replace(reservation_entry, claim_ref=tampered_claim_ref),
        media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    )
    settlement_claim = graph.store.get_json(
        usage.skill_probe_settlement_refs[0],
        SkillProbeUsageSettlementClaim,
    )
    tampered_settlement_ref = graph.store.put_json(
        _replace(settlement_claim, reservation_claim_ref=tampered_claim_ref),
        media_type=SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
    )
    tampered_tail = graph.store.put_json(
        _replace(
            settlement_entry,
            claim_ref=tampered_settlement_ref,
            previous_entry_ref=tampered_reservation_tail,
        ),
        media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    )

    with pytest.raises(ExperimentControllerError, match="schedule"):
        controller.query_usage(tampered_tail)
    assert controller.current_usage() == usage


def test_probe_usage_survives_candidate_lifecycle_advance_and_replays_at_gate_boundary(
    tmp_path: Path,
) -> None:
    executable = _budgeted_executable(
        tmp_path,
        max_evaluations=8,
        max_tokens=2_048,
    )
    graph = executable.graph
    controller, authorization_ref, revert_backend, placebo_backend = _authorized_execution(
        executable
    )
    execution = controller.execute_matched_skill_probes(
        authorization_ref=authorization_ref,
        model_spec=executable.spec,
        revert_backend=revert_backend,
        placebo_backend=placebo_backend,
    )
    controller.verify_matched_skill_probe_result(execution)
    charged = controller.current_usage()
    claim = graph.store.get_json(charged.skill_probe_claim_refs[0], SkillProbeUsageClaim)

    source_ref = _put(
        graph.store,
        {"probe": "score-free matched execution completed"},
        "application/vnd.spiral-harness.probe-source.v1+json",
    )
    evidence = graph.mechanism_service.create(
        protocol_ref=graph.protocol_ref,
        protocol=graph.protocol,
        candidate_ref=graph.candidate_ref,
        candidate_harness_ref=graph.child_ref,
        source_refs=(source_ref,),
        evidence=MechanismEvidence(
            candidate_harness_id=graph.child_ref.sha256,
            checks=(),
        ),
    )
    evidence_ref = _put(
        graph.store,
        evidence,
        ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )
    rejected_tail = controller.start_gate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=claim.running_probes_tail_ref,
        mechanism_evidence_ref=evidence_ref,
    )

    assert (
        CandidateJournal(graph.store).replay(rejected_tail)[-1].to_state is CandidateState.REJECTED
    )
    assert controller.current_usage() == charged
    assert controller.query_usage(charged.tail_ref) == charged


def test_fresh_controller_replays_probe_usage_tail_without_live_authorization_state(
    tmp_path: Path,
) -> None:
    executable = _budgeted_executable(
        tmp_path,
        max_evaluations=8,
        max_tokens=2_048,
    )
    graph = executable.graph
    controller, authorization_ref, revert_backend, placebo_backend = _authorized_execution(
        executable
    )
    controller.execute_matched_skill_probes(
        authorization_ref=authorization_ref,
        model_spec=executable.spec,
        revert_backend=revert_backend,
        placebo_backend=placebo_backend,
    )
    charged = controller.current_usage()
    assert charged.tail_ref is not None

    restarted = ExperimentController(
        graph.store,
        experiment_ref=graph.experiment_ref,
        gate_batch_verifier=graph.gate_batch_service.verification_capability,
        mechanism_evidence_verifier=graph.mechanism_service.verification_capability,
        usage_tail_ref=charged.tail_ref,
    )

    assert restarted.current_usage() == charged
    assert restarted.query_usage(charged.tail_ref) == charged
