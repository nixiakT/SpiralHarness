from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.power_analysis import (
    PowerSensitivityConfig,
    PowerSensitivityReport,
    PrimaryHypothesis,
    SensitivityPurpose,
    run_power_sensitivity,
)
from spiral_harness.experiments.power_joint import gaussian_copula_resample
from spiral_harness.experiments.power_pilot import (
    DeclaredPowerPilotManifest,
    PilotEndpointAssumptionRef,
    bind_declared_power_pilot_manifest,
)
from tools.run_power_sensitivity import main as run_power_sensitivity_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "power_analysis" / "deterministic.json"


def fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def fixture_config() -> PowerSensitivityConfig:
    return PowerSensitivityConfig.model_validate_json(FIXTURE_PATH.read_bytes(), strict=True)


@lru_cache(maxsize=1)
def fixture_report() -> PowerSensitivityReport:
    return run_power_sensitivity(fixture_config())


def validate_payload(payload: dict[str, object]) -> PowerSensitivityConfig:
    return PowerSensitivityConfig.model_validate_json(json.dumps(payload), strict=True)


def artifact(digit: str) -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=1, media_type="application/json")


def correlation(left: list[int], right: list[int]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    return numerator / math.sqrt(left_ss * right_ss)


def declared_manifest(config: PowerSensitivityConfig) -> DeclaredPowerPilotManifest:
    return DeclaredPowerPilotManifest(
        study_id=config.study_id,
        source_kind="declared-power-pilot-metadata",
        declared_synthetic=False,
        declared_partition="disjoint-pilot",
        partition_ref=artifact("a"),
        closure_ref=artifact("b"),
        declared_closure_status="closed-before-sensitivity-run",
        code_revision_sha256="c" * 64,
        endpoint_assumption_refs=tuple(
            PilotEndpointAssumptionRef(
                hypothesis=item.hypothesis.value,
                artifact_ref=ArtifactRef(
                    sha256=item.assumption_artifact_sha256,
                    size=1,
                    media_type="application/json",
                ),
            )
            for item in config.endpoint_assumptions
        ),
        joint_dependence_ref=ArtifactRef(
            sha256=config.joint_primary_assumptions.assumption_artifact_sha256,
            size=1,
            media_type="application/json",
        ),
        declared_main_search_outcomes_visible=False,
    )


def declared_config_and_manifest() -> tuple[PowerSensitivityConfig, bytes]:
    payload = fixture_payload()
    payload["study_id"] = "declared-pilot-digest-study"
    payload["purpose"] = "analytic-sensitivity"
    payload["sensitivity_scenario_label"] = "declared-pilot-plug-in-scenario"
    joint = payload["joint_primary_assumptions"]
    assert isinstance(joint, dict)
    joint["source_label"] = "caller-declared-pilot"
    unbound = validate_payload(payload)
    manifest_payload = canonical_json_bytes(declared_manifest(unbound))
    payload["declared_pilot_manifest_expectation"] = {
        "manifest_sha256": sha256_bytes(manifest_payload),
        "partition_ref_sha256": "a" * 64,
        "closure_ref_sha256": "b" * 64,
        "code_revision_sha256": "c" * 64,
    }
    return validate_payload(payload), manifest_payload


def test_fixture_is_explicit_sensitivity_not_a_design_freeze() -> None:
    config = fixture_config()

    assert config.schema_version == "4"
    assert config.purpose is SensitivityPurpose.DETERMINISTIC_TEST_FIXTURE
    assert config.analysis_method == "bounded-equal-cell-rank-copula-analytic-proxy"
    assert config.real_cell_weighting == "equal-model-by-task-cell"
    assert config.verified_repair_cell_weighting == "equal-model-by-repair-stratum-cell"
    assert tuple(item.hypothesis for item in config.endpoint_assumptions) == (
        PrimaryHypothesis.FULL_VS_SCORE_REAL,
        PrimaryHypothesis.FULL_VS_STATIC_REAL,
        PrimaryHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR,
    )
    assert config.endpoint_assumptions[2].effect_threshold == 0.15
    assert config.project_display_margin == 0.10


def test_report_is_deterministic_and_can_never_freeze_formal_s() -> None:
    config = fixture_config()
    first = fixture_report()
    second = run_power_sensitivity(config)

    assert first == second
    assert first.config_sha256 == canonical_sha256(config)
    assert first.analysis_status == "analytic-sensitivity-proxy-only"
    assert first.formal_required_search_seeds_per_cell is None
    assert first.formal_design_ready is False
    assert first.primary_hierarchical_pipeline_design_ready is False
    assert first.exact_nested_bootstrap_implemented is False
    assert first.formal_fwer_or_power_coverage_claimed is False
    assert first.margin_confidence_support_claimed is False
    assert first.project_display_0_10_is_verified_repair_primary_margin is False
    assert "cannot freeze a formal S" in first.approximation_warning
    assert "independent conditional" in first.audit_warning
    serialized = json.dumps(first.model_dump(mode="json"), sort_keys=True)
    assert "lower_bound" not in serialized
    assert "upper_bound" not in serialized


def test_formal_design_freeze_relabel_is_rejected_by_schema() -> None:
    payload = fixture_payload()
    payload["purpose"] = "design-freeze"

    with pytest.raises(ValidationError, match="purpose"):
        validate_payload(payload)


def test_calibration_audit_and_design_streams_are_separate_and_labeled_proxy() -> None:
    report = fixture_report()

    for diagnostic in report.candidate_diagnostics:
        assert diagnostic.calibration_null_replicates == 600
        assert diagnostic.audit_null_replicates == 700
        assert diagnostic.design_effect_replicates == 800
        assert diagnostic.audit_null_proxy_holm_fwer_rate.replicates == 700
        assert diagnostic.independent_audit_bernoulli_replicates == 700
        assert diagnostic.proxy_all_three_holm_rejection_rate_at_sesoi.replicates == 800
        assert diagnostic.formal_sample_size_candidate is False
        assert (
            diagnostic.proxy_headline_pass_rate_at_sesoi.estimate
            <= diagnostic.proxy_all_three_holm_rejection_rate_at_sesoi.estimate
        )


def test_complete_search_is_only_n_and_h3_weighting_is_explicit() -> None:
    config = fixture_config()
    for diagnostic in fixture_report().candidate_diagnostics:
        search_seeds = diagnostic.search_seeds_per_model_task_cell
        assert diagnostic.independent_unit == "complete_end_to_end_search_run"
        assert diagnostic.real_model_task_cells == 12
        assert diagnostic.repair_model_stratum_cells == 6
        assert diagnostic.real_grid_condition_run_slots == 3 * 4 * 5 * search_seeds
        assert diagnostic.verified_repair_cell_weighting == (config.verified_repair_cell_weighting)


def test_bounded_generation_and_target_bias_are_reported() -> None:
    for diagnostic in fixture_report().candidate_diagnostics:
        boundary = diagnostic.bounded_generation
        assert boundary.method == "winsorized-gaussian-random-effects-proxy"
        assert boundary.support == "[-1,1]"
        assert 0.0 <= boundary.calibration_null_clipping_rate <= 1.0
        assert 0.0 <= boundary.audit_null_clipping_rate <= 1.0
        assert 0.0 <= boundary.design_effect_clipping_rate <= 1.0
        for endpoint in diagnostic.endpoint_diagnostics:
            assert -1.0 <= endpoint.proxy_mean_estimate_at_design_effect <= 1.0
            assert endpoint.proxy_holm_rejection_rate_at_sesoi.standard_error >= 0.0
            assert endpoint.point_estimate_exceeds_sesoi_rate.standard_error >= 0.0
            assert endpoint.point_estimate_exceeds_project_display_0_10_rate.replicates == 800


def test_copula_is_latent_rank_sensitivity_not_claimed_covariance() -> None:
    config = fixture_config()
    size = 4_000
    samples = (list(range(size)), list(range(size)), list(range(size)))

    first, second, repair = gaussian_copula_resample(
        samples,
        assumptions=config.joint_primary_assumptions,
        seed=991,
        key=float,
    )

    assert config.joint_primary_assumptions.correlation_scale == (
        "latent-gaussian-rank-copula-parameter"
    )
    assert correlation(first, second) == pytest.approx(0.58, abs=0.05)
    assert correlation(first, repair) == pytest.approx(0.19, abs=0.05)
    assert "not an exact covariance" in fixture_report().dependence_warning


def test_joint_matrix_must_be_symmetric_psd_and_include_repair() -> None:
    payload = fixture_payload()
    joint = payload["joint_primary_assumptions"]
    assert isinstance(joint, dict)
    joint["correlation_matrix"][1][0] = 0.5
    with pytest.raises(ValidationError, match="symmetric"):
        validate_payload(payload)

    payload = fixture_payload()
    joint = payload["joint_primary_assumptions"]
    assert isinstance(joint, dict)
    joint["correlation_matrix"] = [
        [1.0, 0.9, 0.9],
        [0.9, 1.0, -0.9],
        [0.9, -0.9, 1.0],
    ]
    with pytest.raises(ValidationError, match="positive semidefinite"):
        validate_payload(payload)

    payload = fixture_payload()
    joint = payload["joint_primary_assumptions"]
    assert isinstance(joint, dict)
    joint["hypothesis_order"] = joint["hypothesis_order"][:2]
    with pytest.raises(ValidationError):
        validate_payload(payload)


def test_failure_inputs_are_plugin_sensitivity_not_unused_confidence_bounds() -> None:
    payload = fixture_payload()
    assumptions = payload["endpoint_assumptions"]
    assert isinstance(assumptions, list)
    assert "bound_confidence_level" not in assumptions[0]
    assert "upper" not in json.dumps(assumptions[0])
    failure = assumptions[0]["failure_scenario"]
    failure["full_only_probability"] = 0.4
    failure["control_only_probability"] = 0.4
    failure["both_probability"] = 0.2
    with pytest.raises(ValidationError, match="sum to less than one"):
        validate_payload(payload)
    assert "not a conservative uncertainty set" in fixture_report().failure_warning


def test_schema_freezes_sesoi_and_rejects_rollout_level_n() -> None:
    payload = fixture_payload()
    payload["search_seed_counts_per_cell"] = [1, 3]
    with pytest.raises(ValidationError, match="complete search runs"):
        validate_payload(payload)

    payload = fixture_payload()
    assumptions = payload["endpoint_assumptions"]
    assert isinstance(assumptions, list)
    assumptions[0]["effect_threshold"] = 0.1
    with pytest.raises(ValidationError, match=r"must remain frozen at 0\.02"):
        validate_payload(payload)

    payload = fixture_payload()
    assumptions = payload["endpoint_assumptions"]
    assert isinstance(assumptions, list)
    assumptions[0]["search_seed_icc"] = 0.7
    assumptions[0]["task_group_icc"] = 0.6
    with pytest.raises(ValidationError, match="task_group_icc"):
        validate_payload(payload)


def test_cli_without_declared_manifest_emits_sensitivity_only_report(tmp_path: Path) -> None:
    output = tmp_path / "sensitivity.json"

    assert run_power_sensitivity_main([str(FIXTURE_PATH), str(output)]) == 0
    report = PowerSensitivityReport.model_validate_json(output.read_bytes(), strict=True)
    assert output.read_bytes() == canonical_json_bytes(report)
    assert report.declared_pilot_manifest_digest_bound is False
    assert report.digest_bound_declared_pilot_manifest is None
    assert report.pilot_artifact_existence_verified is False
    assert report.pilot_artifact_authenticity_verified is False
    assert report.pilot_disjointness_verified is False
    assert report.pilot_closure_verified is False
    assert report.formal_design_ready is False

    with pytest.raises(SystemExit) as exc_info:
        run_power_sensitivity_main([str(FIXTURE_PATH), str(output)])
    assert exc_info.value.code == 2


def test_declared_pilot_manifest_digest_binding_does_not_grant_readiness(
    tmp_path: Path,
) -> None:
    config, manifest_payload = declared_config_and_manifest()
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "pilot.json"
    output = tmp_path / "report.json"
    config_path.write_bytes(canonical_json_bytes(config))
    manifest_path.write_bytes(manifest_payload)

    assert (
        run_power_sensitivity_main(
            [
                str(config_path),
                str(output),
                "--pilot-manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    report = PowerSensitivityReport.model_validate_json(output.read_bytes(), strict=True)
    assert report.declared_pilot_manifest_digest_bound is True
    binding = report.digest_bound_declared_pilot_manifest
    assert binding is not None
    assert binding.closure_ref.sha256 == "b" * 64
    assert binding.code_revision_sha256 == "c" * 64
    assert binding.artifact_existence_verified is False
    assert binding.artifact_content_digests_verified is False
    assert binding.manifest_authenticity_verified is False
    assert binding.observed_pilot_status_verified is False
    assert binding.pilot_disjointness_verified is False
    assert binding.pilot_closure_verified is False
    assert "not verified" in report.pilot_provenance_warning
    assert report.formal_design_ready is False
    assert report.formal_required_search_seeds_per_cell is None


def test_declared_manifest_rejects_synthetic_marker_and_digest_mismatch() -> None:
    config, manifest_payload = declared_config_and_manifest()
    values = json.loads(manifest_payload)
    values["declared_synthetic"] = True
    with pytest.raises(ValidationError, match="synthetic"):
        DeclaredPowerPilotManifest.model_validate_json(json.dumps(values), strict=True)

    values = json.loads(manifest_payload)
    values["study_id"] = "synthetic-declared-pilot"
    with pytest.raises(ValidationError, match="synthetic marker"):
        DeclaredPowerPilotManifest.model_validate_json(json.dumps(values), strict=True)

    with pytest.raises(ValueError, match="digest does not match"):
        bind_declared_power_pilot_manifest(
            manifest_payload,
            expectation=config.declared_pilot_manifest_expectation.model_copy(
                update={"manifest_sha256": "0" * 64}
            ),
            expected_study_id=config.study_id,
            expected_endpoint_assumption_hashes=tuple(
                item.assumption_artifact_sha256 for item in config.endpoint_assumptions
            ),
            expected_joint_assumption_hash=(
                config.joint_primary_assumptions.assumption_artifact_sha256
            ),
        )

    wrong_revision = config.declared_pilot_manifest_expectation.model_copy(
        update={"code_revision_sha256": "d" * 64}
    )
    with pytest.raises(ValueError, match="code revision"):
        bind_declared_power_pilot_manifest(
            manifest_payload,
            expectation=wrong_revision,
            expected_study_id=config.study_id,
            expected_endpoint_assumption_hashes=tuple(
                item.assumption_artifact_sha256 for item in config.endpoint_assumptions
            ),
            expected_joint_assumption_hash=(
                config.joint_primary_assumptions.assumption_artifact_sha256
            ),
        )


def test_cli_requires_separate_bound_pilot_artifact(tmp_path: Path) -> None:
    config, _ = declared_config_and_manifest()
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical_json_bytes(config))

    with pytest.raises(SystemExit) as exc_info:
        run_power_sensitivity_main([str(config_path), str(tmp_path / "out.json")])
    assert exc_info.value.code == 2

    with pytest.raises(SystemExit) as exc_info:
        run_power_sensitivity_main(
            [
                str(FIXTURE_PATH),
                str(tmp_path / "other.json"),
                "--pilot-manifest",
                str(tmp_path / "missing.json"),
            ]
        )
    assert exc_info.value.code == 2


def test_run_revalidates_models_and_internally_binds_manifest_bytes() -> None:
    invalid = fixture_config().model_copy(update={"calibration_null_replicates": 2})
    with pytest.raises(ValidationError):
        run_power_sensitivity(invalid)

    config, manifest_payload = declared_config_and_manifest()
    with pytest.raises(ValueError, match="manifest bytes are required"):
        run_power_sensitivity(config)
    with pytest.raises(ValidationError):
        run_power_sensitivity(config, declared_pilot_manifest_payload=b"{}")

    report = run_power_sensitivity(
        config,
        declared_pilot_manifest_payload=manifest_payload,
    )
    assert report.declared_pilot_manifest_digest_bound is True


def test_fixture_cannot_bind_declared_pilot_metadata() -> None:
    payload = fixture_payload()
    payload["declared_pilot_manifest_expectation"] = {
        "manifest_sha256": "a" * 64,
        "partition_ref_sha256": "b" * 64,
        "closure_ref_sha256": "c" * 64,
        "code_revision_sha256": "d" * 64,
    }

    with pytest.raises(ValidationError, match="test fixture cannot bind"):
        validate_payload(payload)


def test_report_rejects_inconsistent_digest_binding_and_config_hash() -> None:
    config, manifest_payload = declared_config_and_manifest()
    report = run_power_sensitivity(
        config,
        declared_pilot_manifest_payload=manifest_payload,
    )
    payload = report.model_dump(mode="json")
    payload["declared_pilot_manifest_digest_bound"] = False
    with pytest.raises(ValidationError, match="digest-bound flag"):
        PowerSensitivityReport.model_validate_json(json.dumps(payload), strict=True)

    payload = report.model_dump(mode="json")
    payload["digest_bound_declared_pilot_manifest"] = None
    with pytest.raises(ValidationError, match="digest-bound flag"):
        PowerSensitivityReport.model_validate_json(json.dumps(payload), strict=True)

    payload = report.model_dump(mode="json")
    payload["config_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="config_sha256"):
        PowerSensitivityReport.model_validate_json(json.dumps(payload), strict=True)
