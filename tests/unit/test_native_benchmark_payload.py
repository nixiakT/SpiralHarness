from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.native_benchmark_payload import (
    BenchmarkRequiredCapability,
    BenchmarkRequiredCapabilityKind,
    FrozenNativeBenchmarkPayload,
    NativeBenchmarkArm,
    NativeBenchmarkArmBinding,
    NativeBenchmarkFiveArmContract,
    NativeEvaluationResourceCeilings,
    NativePureAtBTotalBudgetMatch,
    NativeTotalResourceCeilings,
    ProviderMinimalNativeTaskPlan,
    make_provider_minimal_native_task_plan,
)


def _ref(character: str, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(sha256=character * 64, size=1, media_type=media_type)


def _native_payload(*, task_character: str = "1") -> FrozenNativeBenchmarkPayload:
    return FrozenNativeBenchmarkPayload(
        benchmark_id="bfcl-v4@6ea57973",
        task_payload_ref=_ref(task_character),
        provider_request_schema_ref=_ref("2"),
        provider_serialization_implementation_ref=_ref("3", "text/x-python"),
        provider_serialization_config_ref=_ref("4"),
        required_capabilities=(
            BenchmarkRequiredCapability(
                capability_id="official-function-schemas",
                kind=BenchmarkRequiredCapabilityKind.TOOL_SCHEMA_BUNDLE,
                definition_ref=_ref("5"),
            ),
            BenchmarkRequiredCapability(
                capability_id="official-function-runtime",
                kind=BenchmarkRequiredCapabilityKind.TOOL_EXECUTION_RUNTIME,
                definition_ref=_ref("6"),
                runtime_implementation_ref=_ref("7", "text/x-python"),
            ),
        ),
    )


def _budget(*, max_tool_executions: int = 20) -> NativeEvaluationResourceCeilings:
    return NativeEvaluationResourceCeilings(
        max_model_steps_per_turn=20,
        max_provider_attempts_per_model_step=2,
        token_ceiling_per_model_step=4096,
        max_tool_executions_per_evaluation_unit=max_tool_executions,
        max_search_queries_per_evaluation_unit=20,
        max_http_fetches_per_evaluation_unit=20,
        max_downloaded_bytes_per_evaluation_unit=5_000_000,
        max_wall_time_seconds_per_evaluation_unit=300.0,
        max_external_cost_usd_per_evaluation_unit=1.0,
        retry_policy_ref=_ref("8"),
        price_table_ref=_ref("9"),
        sandbox_policy_ref=_ref("a"),
    )


def _arm_bindings(
    payload: FrozenNativeBenchmarkPayload,
    budget: NativeEvaluationResourceCeilings,
) -> tuple[NativeBenchmarkArmBinding, ...]:
    minimal = make_provider_minimal_native_task_plan(payload)
    return tuple(
        NativeBenchmarkArmBinding(
            arm=arm,
            native_payload_fingerprint=payload.fingerprint,
            evaluation_resource_ceilings=budget,
            provider_minimal_plan=(
                minimal if arm in {NativeBenchmarkArm.PURE, NativeBenchmarkArm.PURE_AT_B} else None
            ),
        )
        for arm in reversed(tuple(NativeBenchmarkArm))
    )


def _total_budget(*, search_queries: int = 100) -> NativeTotalResourceCeilings:
    return NativeTotalResourceCeilings(
        max_model_steps=1000,
        max_provider_attempts=2000,
        max_provider_attempt_tokens=8_192_000,
        max_tool_executions=1000,
        max_search_queries=search_queries,
        max_http_fetches=100,
        max_downloaded_bytes=50_000_000,
        max_wall_time_seconds=10_000.0,
        max_external_cost_usd=100.0,
    )


def _pure_at_b_budget_match() -> NativePureAtBTotalBudgetMatch:
    totals = _total_budget()
    return NativePureAtBTotalBudgetMatch(
        full_total_ceilings=totals,
        pure_at_b_total_ceilings=totals,
        model_budget_plan_ref=_ref("e"),
        external_resource_allocation_ref=_ref("0"),
    )


def _five_arm_contract(
    *,
    payload: FrozenNativeBenchmarkPayload | None = None,
    budget: NativeEvaluationResourceCeilings | None = None,
    arms: tuple[NativeBenchmarkArmBinding, ...] | None = None,
) -> NativeBenchmarkFiveArmContract:
    checked_payload = payload or _native_payload()
    checked_budget = budget or _budget()
    return NativeBenchmarkFiveArmContract(
        benchmark_id="bfcl-v4@6ea57973",
        public_snapshot_fingerprint="b" * 64,
        public_roster_fingerprint="c" * 64,
        confirmatory_four_arm_design_ref=_ref("d"),
        confirmatory_pure_at_b_plan_ref=_ref("e"),
        score_full_adaptive_budget_ref=_ref("f"),
        pure_at_b_total_budget_match=_pure_at_b_budget_match(),
        arms=arms or _arm_bindings(checked_payload, checked_budget),
    )


def test_executable_native_capability_requires_runtime_ref() -> None:
    with pytest.raises(ValidationError, match="require exactly one runtime ref"):
        BenchmarkRequiredCapability(
            capability_id="runtime",
            kind=BenchmarkRequiredCapabilityKind.TOOL_EXECUTION_RUNTIME,
            definition_ref=_ref("1"),
        )

    with pytest.raises(ValidationError, match="require exactly one runtime ref"):
        BenchmarkRequiredCapability(
            capability_id="schemas",
            kind=BenchmarkRequiredCapabilityKind.TOOL_SCHEMA_BUNDLE,
            definition_ref=_ref("1"),
            runtime_implementation_ref=_ref("2"),
        )


def test_native_payload_canonicalizes_capabilities_and_rejects_aliasing() -> None:
    payload = _native_payload()
    assert tuple(item.capability_id for item in payload.required_capabilities) == (
        "official-function-runtime",
        "official-function-schemas",
    )

    duplicated = payload.required_capabilities[0].model_copy(
        update={"capability_id": "another-runtime"}
    )
    with pytest.raises(ValidationError, match="definition refs must not repeat"):
        FrozenNativeBenchmarkPayload(
            benchmark_id="bfcl-v4@6ea57973",
            task_payload_ref=_ref("1"),
            provider_request_schema_ref=_ref("2"),
            provider_serialization_implementation_ref=_ref("3"),
            provider_serialization_config_ref=_ref("4"),
            required_capabilities=(payload.required_capabilities[0], duplicated),
        )


def test_native_payload_fingerprint_binds_serializer_schema_config_and_runtime() -> None:
    payload = _native_payload()
    changed_top_level = tuple(
        FrozenNativeBenchmarkPayload.model_validate(
            {
                **payload.model_dump(mode="python"),
                field: _ref(character, media_type).model_dump(mode="python"),
            },
            strict=True,
        )
        for field, character, media_type in (
            ("provider_request_schema_ref", "8", "application/json"),
            ("provider_serialization_implementation_ref", "9", "text/x-python"),
            ("provider_serialization_config_ref", "a", "application/json"),
        )
    )
    runtime, schemas = payload.required_capabilities
    changed_runtime = runtime.model_copy(update={"runtime_implementation_ref": _ref("b")})
    changed_capability = FrozenNativeBenchmarkPayload.model_validate(
        {
            **payload.model_dump(mode="python"),
            "required_capabilities": (
                changed_runtime.model_dump(mode="python"),
                schemas.model_dump(mode="python"),
            ),
        },
        strict=True,
    )

    fingerprints = {
        payload.fingerprint,
        *(candidate.fingerprint for candidate in changed_top_level),
        changed_capability.fingerprint,
    }
    assert len(fingerprints) == 5


def test_provider_minimal_plan_preserves_official_tools_but_forbids_harness_additions() -> None:
    payload = _native_payload()
    plan = make_provider_minimal_native_task_plan(payload)
    assert plan.official_task_capabilities_preserved is True
    assert plan.benchmark_required_tools_are_task_payload is True
    assert plan.harness_added_capabilities_permitted is False
    assert plan.harness_system_prompt_ref is None
    assert plan.harness_tool_bundle_ref is None
    assert plan.harness_routing_ref is None
    assert plan.harness_middleware_ref is None

    with pytest.raises(ValidationError):
        ProviderMinimalNativeTaskPlan.model_validate(
            {
                **plan.model_dump(mode="python"),
                "harness_system_prompt_ref": _ref("a").model_dump(mode="python"),
            },
            strict=True,
        )


def test_provider_minimal_plan_revalidates_before_publication() -> None:
    plan = make_provider_minimal_native_task_plan(_native_payload())
    object.__setattr__(plan, "harness_tool_bundle_ref", _ref("a"))
    with pytest.raises(ValidationError):
        plan.model_dump(mode="python")


def test_five_arm_contract_canonicalizes_and_matches_native_payload_and_budget() -> None:
    contract = _five_arm_contract()
    assert tuple(item.arm for item in contract.arms) == tuple(NativeBenchmarkArm)
    assert contract.public_answers_are_hidden_evidence is False
    assert contract.partial_evaluation_permitted is False
    assert contract.reportable_result is False
    assert contract.arms[0].provider_minimal_plan == contract.arms[-1].provider_minimal_plan


def test_five_arm_contract_rejects_native_payload_drift() -> None:
    payload = _native_payload()
    arms = list(_arm_bindings(payload, _budget()))
    arms[1] = arms[1].model_copy(
        update={"native_payload_fingerprint": _native_payload(task_character="0").fingerprint}
    )
    with pytest.raises(ValidationError, match="one frozen native task payload"):
        _five_arm_contract(payload=payload, arms=tuple(arms))


def test_five_arm_contract_rejects_benchmark_id_join_drift() -> None:
    contract = _five_arm_contract()
    with pytest.raises(ValidationError, match="native payload benchmark IDs differ"):
        NativeBenchmarkFiveArmContract.model_validate(
            {
                **contract.model_dump(mode="python"),
                "benchmark_id": "different-benchmark@revision",
            },
            strict=True,
        )


def test_five_arm_contract_rejects_pure_at_b_plan_join_drift() -> None:
    contract = _five_arm_contract()
    with pytest.raises(ValidationError, match="total-budget match bind different plans"):
        NativeBenchmarkFiveArmContract.model_validate(
            {
                **contract.model_dump(mode="python"),
                "confirmatory_pure_at_b_plan_ref": _ref("1").model_dump(mode="python"),
            },
            strict=True,
        )


def test_five_arm_contract_rejects_official_evaluation_budget_drift() -> None:
    payload = _native_payload()
    arms = list(_arm_bindings(payload, _budget()))
    arms[2] = arms[2].model_copy(
        update={"evaluation_resource_ceilings": _budget(max_tool_executions=19)}
    )
    with pytest.raises(ValidationError, match="official evaluation resource ceilings"):
        _five_arm_contract(payload=payload, arms=tuple(arms))


def test_only_pure_and_pure_at_b_accept_provider_minimal_plans() -> None:
    payload = _native_payload()
    minimal = make_provider_minimal_native_task_plan(payload)
    with pytest.raises(ValidationError, match="only PURE and PURE@B"):
        NativeBenchmarkArmBinding(
            arm=NativeBenchmarkArm.STATIC,
            native_payload_fingerprint=payload.fingerprint,
            evaluation_resource_ceilings=_budget(),
            provider_minimal_plan=minimal,
        )


def test_pure_at_b_native_total_budget_matches_external_resources_too() -> None:
    with pytest.raises(ValidationError, match="native total resources differ"):
        NativePureAtBTotalBudgetMatch(
            full_total_ceilings=_total_budget(),
            pure_at_b_total_ceilings=_total_budget(search_queries=101),
            model_budget_plan_ref=_ref("1"),
            external_resource_allocation_ref=_ref("2"),
        )


def test_five_arm_contract_cannot_upgrade_public_answers_to_hidden() -> None:
    contract = _five_arm_contract()
    with pytest.raises(ValidationError):
        NativeBenchmarkFiveArmContract.model_validate(
            {
                **contract.model_dump(mode="python"),
                "public_answers_are_hidden_evidence": True,
            },
            strict=True,
        )
