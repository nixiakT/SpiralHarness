"""Verified artifact boundary for the non-reportable confirmatory rehearsal.

The functions here perform no execution.  They reload every typed plan input,
read every opaque content reference, and fail closed on missing bytes, digest,
size, media-type, canonical-JSON, roster, or expected-plan drift.
"""

from __future__ import annotations

from pydantic import BaseModel, TypeAdapter

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    HarnessManifest,
)
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.experiments.confirmatory_arms import (
    CONFIRMATORY_FOUR_ARM_DESIGN_MEDIA_TYPE,
    PURE_AT_B_PLAN_MEDIA_TYPE,
    ConfirmatoryFourArmDesign,
    RealTaskArm,
)
from spiral_harness.experiments.confirmatory_execution_contracts import (
    CONFIRMATORY_FROZEN_MODEL_SPEC_MEDIA_TYPE,
    CONFIRMATORY_REHEARSAL_PLAN_MEDIA_TYPE,
    ConfirmatoryRehearsalPlan,
)
from spiral_harness.experiments.confirmatory_pure_at_b import PureAtBPlan
from spiral_harness.experiments.confirmatory_resources import (
    CONFIRMATORY_MUTATION_POLICY_MEDIA_TYPE,
    CONFIRMATORY_TASK_SPLIT_MEDIA_TYPE,
    ConfirmatoryTaskSplitManifest,
    FrozenMutationPolicyArtifact,
)
from spiral_harness.storage.protocol import ArtifactRepository


class ConfirmatoryRehearsalArtifactError(ValueError):
    """A rehearsal artifact or one of its transitive inputs is not exact."""


def _require_repository(repository: ArtifactRepository) -> ArtifactRepository:
    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    return repository


def _require_ref_media(ref: ArtifactRef, media_type: str, label: str) -> ArtifactRef:
    checked = ArtifactRef.model_validate(ref, strict=True)
    if checked.media_type != media_type:
        raise ConfirmatoryRehearsalArtifactError(f"{label} ref declares the wrong media type")
    return checked


def _read_exact(
    repository: ArtifactRepository,
    ref: ArtifactRef,
    label: str,
) -> bytes:
    checked = ArtifactRef.model_validate(ref, strict=True)
    try:
        payload = bytes(repository.get_bytes(checked))
    except Exception as exc:
        raise ConfirmatoryRehearsalArtifactError(f"{label} content cannot be loaded") from exc
    if len(payload) != checked.size:
        raise ConfirmatoryRehearsalArtifactError(f"{label} content size differs from its ref")
    if sha256_bytes(payload) != checked.sha256:
        raise ConfirmatoryRehearsalArtifactError(f"{label} content digest differs from its ref")
    return payload


def _load_typed[ModelT: BaseModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    media_type: str,
    label: str,
) -> ModelT:
    checked_ref = _require_ref_media(ref, media_type, label)
    payload = _read_exact(repository, checked_ref, label)
    try:
        loaded = TypeAdapter(model_type).validate_json(payload, strict=True)
        checked = model_type.model_validate(loaded, strict=True)
        canonical = canonical_json_bytes(checked)
    except Exception as exc:
        raise ConfirmatoryRehearsalArtifactError(
            f"{label} content is not strict canonical typed JSON"
        ) from exc
    if canonical != payload:
        raise ConfirmatoryRehearsalArtifactError(
            f"{label} content differs from its canonical typed representation"
        )
    computed = ArtifactRef(
        sha256=sha256_bytes(canonical),
        size=len(canonical),
        media_type=media_type,
    )
    if computed != checked_ref:
        raise ConfirmatoryRehearsalArtifactError(
            f"{label} canonical content differs from its complete ref"
        )
    return checked


def _ref_key(ref: ArtifactRef) -> tuple[str, int, str]:
    return (ref.sha256, ref.size, ref.media_type)


def _reject_conflicting_refs(refs: tuple[tuple[str, ArtifactRef], ...]) -> None:
    by_digest: dict[str, tuple[int, str, str]] = {}
    for label, ref in refs:
        checked = ArtifactRef.model_validate(ref, strict=True)
        prior = by_digest.get(checked.sha256)
        identity = (checked.size, checked.media_type, label)
        if prior is not None and prior[:2] != identity[:2]:
            raise ConfirmatoryRehearsalArtifactError(
                f"content digest is declared with conflicting refs: {prior[2]} and {label}"
            )
        by_digest[checked.sha256] = identity


def _plan_refs(
    plan: ConfirmatoryRehearsalPlan,
    mutation_policy: FrozenMutationPolicyArtifact,
) -> tuple[tuple[str, ArtifactRef], ...]:
    full = plan.four_arm_design.arm(RealTaskArm.FULL)
    topology = full.adaptive_topology
    if topology is None:  # pragma: no cover - strict design invariant
        raise ConfirmatoryRehearsalArtifactError("FULL adaptive topology is missing")
    commitments = topology.protocol_commitments
    refs: list[tuple[str, ArtifactRef]] = [
        ("preregistration", plan.preregistration_ref),
        ("four-arm design", plan.four_arm_design_ref),
        ("PURE@B plan", plan.pure_at_b_plan_ref),
        ("frozen model spec", plan.model_spec_ref),
        ("seed harness", plan.seed_harness_ref),
        ("task split", commitments.task_split_manifest_ref),
        ("mutation policy", commitments.mutation_policy_ref),
        ("solver config", plan.solver_config_ref),
        ("optimizer config", plan.optimizer_config_ref),
        ("candidate parser", plan.candidate_parser_implementation_ref),
    ]
    for component in plan.seed_harness.components:
        refs.append((f"seed harness component {component.name}", component.artifact))
    for task in plan.task_bindings:
        refs.append((f"task manifest {task.task_id}", task.task_manifest_ref))
        refs.extend(
            (
                f"evaluation unit {unit.evaluation_unit_id}",
                unit.evaluation_unit_ref,
            )
            for unit in task.evaluation_units
        )
        for field_name in (
            "adapter_implementation_ref",
            "grader_implementation_ref",
            "query_dag_implementation_ref",
            "retry_policy_implementation_ref",
            "price_table_ref",
            "aggregation_implementation_ref",
            "aggregation_normalizer_ref",
            "aggregation_output_domain_ref",
        ):
            refs.append(
                (
                    f"{task.task_id} {field_name}",
                    getattr(task, field_name),
                )
            )
    for surface in mutation_policy.surface_grammars:
        prefix = f"mutation surface {surface.component_kind.value}"
        refs.extend(
            (
                (f"{prefix} grammar", surface.grammar_ref),
                (f"{prefix} candidate schema", surface.candidate_schema_ref),
                (f"{prefix} parser", surface.parser_implementation_ref),
                (f"{prefix} materializer", surface.materializer_implementation_ref),
            )
        )
        refs.extend(
            (f"{prefix} seed component {component.component_name}", component.artifact_ref)
            for component in surface.seed_components
        )
    refs.append(("mutation policy provenance", mutation_policy.construction_provenance_ref))
    return tuple(refs)


def _verify_bound_artifacts(
    repository: ArtifactRepository,
    plan: ConfirmatoryRehearsalPlan,
) -> None:
    design = _load_typed(
        repository,
        plan.four_arm_design_ref,
        ConfirmatoryFourArmDesign,
        CONFIRMATORY_FOUR_ARM_DESIGN_MEDIA_TYPE,
        "four-arm design",
    )
    if design != plan.four_arm_design:
        raise ConfirmatoryRehearsalArtifactError("loaded four-arm design drifted from rehearsal")
    pure_at_b = _load_typed(
        repository,
        plan.pure_at_b_plan_ref,
        PureAtBPlan,
        PURE_AT_B_PLAN_MEDIA_TYPE,
        "PURE@B plan",
    )
    if pure_at_b != plan.pure_at_b_plan:
        raise ConfirmatoryRehearsalArtifactError("loaded PURE@B plan drifted from rehearsal")
    model_spec = _load_typed(
        repository,
        plan.model_spec_ref,
        FrozenModelSpec,
        CONFIRMATORY_FROZEN_MODEL_SPEC_MEDIA_TYPE,
        "frozen model spec",
    )
    if model_spec != plan.model_spec:
        raise ConfirmatoryRehearsalArtifactError("loaded model spec drifted from rehearsal")
    seed_harness = _load_typed(
        repository,
        plan.seed_harness_ref,
        HarnessManifest,
        HARNESS_MANIFEST_MEDIA_TYPE,
        "seed harness",
    )
    if seed_harness != plan.seed_harness:
        raise ConfirmatoryRehearsalArtifactError("loaded seed harness drifted from rehearsal")

    full = design.arm(RealTaskArm.FULL)
    topology = full.adaptive_topology
    if topology is None:  # pragma: no cover - strict design invariant
        raise ConfirmatoryRehearsalArtifactError("FULL adaptive topology is missing")
    commitments = topology.protocol_commitments
    task_split = _load_typed(
        repository,
        commitments.task_split_manifest_ref,
        ConfirmatoryTaskSplitManifest,
        CONFIRMATORY_TASK_SPLIT_MEDIA_TYPE,
        "task split",
    )
    if task_split != commitments.task_split_manifest:
        raise ConfirmatoryRehearsalArtifactError("loaded task split drifted from rehearsal")
    mutation_policy = _load_typed(
        repository,
        commitments.mutation_policy_ref,
        FrozenMutationPolicyArtifact,
        CONFIRMATORY_MUTATION_POLICY_MEDIA_TYPE,
        "mutation policy",
    )
    if mutation_policy.artifact_ref != commitments.mutation_policy_ref:
        raise ConfirmatoryRehearsalArtifactError("loaded mutation policy ref is not canonical")
    if mutation_policy.seed_harness_ref != plan.seed_harness_ref:
        raise ConfirmatoryRehearsalArtifactError("mutation policy binds a different seed harness")

    refs = _plan_refs(plan, mutation_policy)
    _reject_conflicting_refs(refs)
    seen: set[tuple[str, int, str]] = {
        _ref_key(plan.four_arm_design_ref),
        _ref_key(plan.pure_at_b_plan_ref),
        _ref_key(plan.model_spec_ref),
        _ref_key(plan.seed_harness_ref),
        _ref_key(commitments.task_split_manifest_ref),
        _ref_key(commitments.mutation_policy_ref),
    }
    for label, ref in refs:
        key = _ref_key(ref)
        if key in seen:
            continue
        _read_exact(repository, ref, label)
        seen.add(key)


def publish_confirmatory_rehearsal_plan(
    repository: ArtifactRepository,
    plan: ConfirmatoryRehearsalPlan,
) -> ArtifactRef:
    """Publish a content-reference-complete rehearsal plan; never execute it."""

    store = _require_repository(repository)
    checked = ConfirmatoryRehearsalPlan.model_validate(plan, strict=True)
    _verify_bound_artifacts(store, checked)
    try:
        ref = store.put_json(checked, media_type=CONFIRMATORY_REHEARSAL_PLAN_MEDIA_TYPE)
    except Exception as exc:
        raise ConfirmatoryRehearsalArtifactError("rehearsal plan cannot be published") from exc
    if ref != checked.artifact_ref:
        raise ConfirmatoryRehearsalArtifactError(
            "published rehearsal ref differs from canonical plan content"
        )
    load_confirmatory_rehearsal_plan(store, ref)
    return ref


def load_confirmatory_rehearsal_plan(
    repository: ArtifactRepository,
    ref: ArtifactRef,
) -> ConfirmatoryRehearsalPlan:
    """Strictly load a rehearsal and all of its transitive content refs."""

    store = _require_repository(repository)
    checked_ref = _require_ref_media(ref, CONFIRMATORY_REHEARSAL_PLAN_MEDIA_TYPE, "rehearsal")
    plan = _load_typed(
        store,
        checked_ref,
        ConfirmatoryRehearsalPlan,
        CONFIRMATORY_REHEARSAL_PLAN_MEDIA_TYPE,
        "rehearsal",
    )
    if plan.artifact_ref != checked_ref:
        raise ConfirmatoryRehearsalArtifactError("rehearsal ref differs from canonical plan")
    _verify_bound_artifacts(store, plan)
    return plan


def verify_confirmatory_rehearsal_plan(
    repository: ArtifactRepository,
    ref: ArtifactRef,
    expected: ConfirmatoryRehearsalPlan,
) -> ConfirmatoryRehearsalPlan:
    """Reload every input and require equality with the frozen expected plan."""

    checked_expected = ConfirmatoryRehearsalPlan.model_validate(expected, strict=True)
    if checked_expected.artifact_ref != ref:
        raise ConfirmatoryRehearsalArtifactError(
            "rehearsal ref differs from the expected canonical plan"
        )
    loaded = load_confirmatory_rehearsal_plan(repository, ref)
    if loaded != checked_expected:  # pragma: no cover - canonical ref equality implies equality
        raise ConfirmatoryRehearsalArtifactError("loaded rehearsal differs from expected plan")
    return loaded


__all__ = [
    "ConfirmatoryRehearsalArtifactError",
    "load_confirmatory_rehearsal_plan",
    "publish_confirmatory_rehearsal_plan",
    "verify_confirmatory_rehearsal_plan",
]
