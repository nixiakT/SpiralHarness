"""Content-reference contracts for a permanently non-reportable, non-attested rehearsal."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    HarnessManifest,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.experiments.confirmatory_arms import (
    CONFIRMATORY_FOUR_ARM_DESIGN_MEDIA_TYPE,
    PURE_AT_B_PLAN_MEDIA_TYPE,
    ConfirmatoryFourArmDesign,
    RealTaskArm,
)
from spiral_harness.experiments.confirmatory_pure_at_b import PureAtBPlan
from spiral_harness.experiments.confirmatory_resources import (
    ModelMediatedRole,
    ProspectiveConfirmatoryModel,
    TaskSplitEvaluationUnit,
)

CONFIRMATORY_FROZEN_MODEL_SPEC_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-frozen-model-spec.v1+json"
)
CONFIRMATORY_REHEARSAL_PLAN_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-rehearsal-plan.v1+json"
)

SearchSeed = Annotated[int, Field(ge=0, le=9_223_372_036_854_775_807, strict=True)]
ImplementationBundleField = Literal[
    "grader_implementation_ref",
    "query_dag_implementation_ref",
    "retry_policy_implementation_ref",
]


def _canonical_ref(value: object, media_type: str) -> ArtifactRef:
    payload = canonical_json_bytes(value)
    return ArtifactRef(
        sha256=canonical_sha256(value),
        size=len(payload),
        media_type=media_type,
    )


class _RehearsalModel(ProspectiveConfirmatoryModel):
    @classmethod
    def model_construct(
        cls,
        _fields_set: set[str] | None = None,
        **values: Any,
    ) -> Self:
        del _fields_set
        return cls.model_validate(values, strict=True)


class ConfirmatoryExecutionCondition(StrEnum):
    PURE = "pure"
    STATIC = "static"
    SCORE = "score"
    FULL = "full"
    PURE_AT_B = "pure-at-b"


_CONDITION_ORDER = (
    ConfirmatoryExecutionCondition.PURE,
    ConfirmatoryExecutionCondition.STATIC,
    ConfirmatoryExecutionCondition.SCORE,
    ConfirmatoryExecutionCondition.FULL,
    ConfirmatoryExecutionCondition.PURE_AT_B,
)


class ConfirmatorySearchSeed(_RehearsalModel):
    schema_version: Literal["1"] = "1"
    seed_id: NonEmptyStr
    seed: SearchSeed

    @field_validator("seed_id", mode="before")
    @classmethod
    def _seed_id_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("search seed ID must be exact and non-empty")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class ConfirmatoryEvaluationUnitSeed(_RehearsalModel):
    schema_version: Literal["1"] = "1"
    evaluation_unit_id: NonEmptyStr
    evaluation_unit_ref: ArtifactRef
    rollout_seed: SearchSeed
    seed_domain: Literal["spiral-harness/confirmatory-evaluation-rollout/v1"] = (
        "spiral-harness/confirmatory-evaluation-rollout/v1"
    )

    @field_validator("evaluation_unit_id", mode="before")
    @classmethod
    def _unit_id_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("evaluation-unit seed ID must be exact and non-empty")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class ConfirmatoryTaskRuntimeBinding(_RehearsalModel):
    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    task_manifest_ref: ArtifactRef
    evaluation_units: Annotated[tuple[TaskSplitEvaluationUnit, ...], Field(min_length=1)]
    evaluation_seeds: Annotated[
        tuple[ConfirmatoryEvaluationUnitSeed, ...],
        Field(min_length=1),
    ]
    adapter_implementation_ref: ArtifactRef
    grader_implementation_ref: ArtifactRef
    query_dag_implementation_ref: ArtifactRef
    retry_policy_implementation_ref: ArtifactRef
    price_table_ref: ArtifactRef
    aggregation_implementation_ref: ArtifactRef
    aggregation_normalizer_ref: ArtifactRef
    aggregation_output_domain_ref: ArtifactRef

    @field_validator("task_id", mode="before")
    @classmethod
    def _task_id_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("task_id must be exact and non-empty")
        return value

    @field_validator("evaluation_units")
    @classmethod
    def _canonicalize_units(
        cls,
        values: tuple[TaskSplitEvaluationUnit, ...],
    ) -> tuple[TaskSplitEvaluationUnit, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.evaluation_unit_id))
        ids = tuple(item.evaluation_unit_id for item in ordered)
        refs = tuple(
            (
                item.evaluation_unit_ref.sha256,
                item.evaluation_unit_ref.size,
                item.evaluation_unit_ref.media_type,
            )
            for item in ordered
        )
        if len(ids) != len(set(ids)):
            raise ValueError("task runtime evaluation-unit IDs must not repeat")
        if len(refs) != len(set(refs)):
            raise ValueError("task runtime evaluation-unit refs must not repeat")
        return ordered

    @field_validator("evaluation_seeds")
    @classmethod
    def _canonicalize_evaluation_seeds(
        cls,
        values: tuple[ConfirmatoryEvaluationUnitSeed, ...],
    ) -> tuple[ConfirmatoryEvaluationUnitSeed, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.evaluation_unit_id))
        ids = tuple(item.evaluation_unit_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation-unit seed coordinates must not repeat")
        rollout_seeds = tuple(item.rollout_seed for item in ordered)
        if len(rollout_seeds) != len(set(rollout_seeds)):
            raise ValueError("evaluation-unit rollout seeds must not repeat")
        return ordered

    @model_validator(mode="after")
    def _bind_every_evaluation_seed(self) -> Self:
        expected = {
            (unit.evaluation_unit_id, unit.evaluation_unit_ref) for unit in self.evaluation_units
        }
        actual = {
            (seed.evaluation_unit_id, seed.evaluation_unit_ref) for seed in self.evaluation_seeds
        }
        if actual != expected or len(actual) != len(self.evaluation_seeds):
            raise ValueError("evaluation seeds are not bijective with the task unit roster")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


_BUNDLE_FIELDS = frozenset(
    {
        "grader_implementation_ref",
        "query_dag_implementation_ref",
        "retry_policy_implementation_ref",
    }
)


def task_implementation_bundle_fingerprint(
    bindings: tuple[ConfirmatoryTaskRuntimeBinding, ...],
    field_name: ImplementationBundleField,
) -> str:
    """Hash an ordered task-to-implementation binding without caller claims."""

    if field_name not in _BUNDLE_FIELDS:  # pragma: no cover - static callers are typed
        raise ValueError("unsupported implementation bundle field")
    checked = tuple(
        ConfirmatoryTaskRuntimeBinding.model_validate(item, strict=True) for item in bindings
    )
    ordered = tuple(sorted(checked, key=lambda item: item.task_id))
    return canonical_sha256(
        {
            "schema_version": "1",
            "field": field_name,
            "tasks": tuple(
                {
                    "task_id": item.task_id,
                    "implementation_ref": getattr(item, field_name),
                }
                for item in ordered
            ),
        }
    )


def confirmatory_search_seed_schedule_fingerprint(
    bindings: tuple[ConfirmatoryTaskRuntimeBinding, ...],
    search_seeds: tuple[ConfirmatorySearchSeed, ...],
) -> str:
    """Derive the paired task-by-search-seed schedule identity."""

    checked = tuple(
        ConfirmatoryTaskRuntimeBinding.model_validate(item, strict=True) for item in bindings
    )
    ordered_tasks = tuple(sorted(checked, key=lambda item: item.task_id))
    if not search_seeds:
        raise ValueError("confirmatory rehearsal requires at least one search seed")
    checked_seeds = tuple(
        ConfirmatorySearchSeed.model_validate(seed, strict=True) for seed in search_seeds
    )
    ordered_seeds = tuple(sorted(checked_seeds, key=lambda item: item.seed_id))
    seed_ids = tuple(item.seed_id for item in ordered_seeds)
    seed_fingerprints = tuple(item.fingerprint for item in ordered_seeds)
    seed_values = tuple(item.seed for item in ordered_seeds)
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("search seed IDs must not repeat")
    if len(seed_fingerprints) != len(set(seed_fingerprints)):
        raise ValueError("full search seeds must not repeat")
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("raw search seed values must not repeat")
    return canonical_sha256(
        {
            "schema_version": "1",
            "tasks": tuple(
                {
                    "task_id": task.task_id,
                    "task_manifest_ref": task.task_manifest_ref,
                    "evaluation_units": task.evaluation_units,
                    "evaluation_seeds": task.evaluation_seeds,
                }
                for task in ordered_tasks
            ),
            "search_seeds": tuple(
                {
                    "seed_id": seed.seed_id,
                    "seed_fingerprint": seed.fingerprint,
                }
                for seed in ordered_seeds
            ),
        }
    )


class ConfirmatoryExecutionKey(_RehearsalModel):
    schema_version: Literal["1"] = "1"
    study_id: NonEmptyStr
    preregistration_ref: ArtifactRef
    four_arm_design_ref: ArtifactRef
    pure_at_b_plan_ref: ArtifactRef
    model_spec_ref: ArtifactRef
    seed_harness_ref: ArtifactRef
    solver_config_ref: ArtifactRef
    optimizer_config_ref: ArtifactRef
    candidate_parser_implementation_ref: ArtifactRef
    condition: ConfirmatoryExecutionCondition
    task_id: NonEmptyStr
    task_manifest_ref: ArtifactRef
    task_runtime_binding_fingerprint: Sha256
    search_seed: ConfirmatorySearchSeed
    matched_block_fingerprint: Sha256
    key_fingerprint: Sha256

    @model_validator(mode="after")
    def _key_fingerprint_is_derived(self) -> Self:
        expected_block = canonical_sha256(
            {
                "domain": "spiral-harness/confirmatory-rehearsal-matched-block/v1",
                "study_id": self.study_id,
                "preregistration_ref": self.preregistration_ref,
                "four_arm_design_ref": self.four_arm_design_ref,
                "pure_at_b_plan_ref": self.pure_at_b_plan_ref,
                "model_spec_ref": self.model_spec_ref,
                "seed_harness_ref": self.seed_harness_ref,
                "solver_config_ref": self.solver_config_ref,
                "optimizer_config_ref": self.optimizer_config_ref,
                "candidate_parser_implementation_ref": (self.candidate_parser_implementation_ref),
                "task_id": self.task_id,
                "task_manifest_ref": self.task_manifest_ref,
                "task_runtime_binding_fingerprint": (self.task_runtime_binding_fingerprint),
                "search_seed_id": self.search_seed.seed_id,
                "search_seed_fingerprint": self.search_seed.fingerprint,
            }
        )
        if self.matched_block_fingerprint != expected_block:
            raise ValueError("matched block fingerprint is not derived from shared coordinates")
        expected_cell = canonical_sha256(
            {
                "domain": "spiral-harness/confirmatory-rehearsal-cell/v1",
                "matched_block_fingerprint": expected_block,
                "condition": self.condition,
            }
        )
        if self.key_fingerprint != expected_cell:
            raise ValueError("execution-key fingerprint is not derived from its coordinate")
        return self


class ConfirmatoryModelRoleBinding(_RehearsalModel):
    schema_version: Literal["1"] = "1"
    role: ModelMediatedRole
    enabled: bool
    model_spec_ref: ArtifactRef | None

    @model_validator(mode="after")
    def _enabled_roles_have_exactly_one_model(self) -> Self:
        if self.enabled != (self.model_spec_ref is not None):
            raise ValueError("only enabled model-mediated roles may bind a model spec")
        return self


class ConfirmatoryRehearsalPlan(_RehearsalModel):
    """Content-ref-closed rehearsal that is permanently development-only."""

    schema_version: Literal["1"] = "1"
    rehearsal_kind: Literal["first-small-batch-non-reportable"] = "first-small-batch-non-reportable"
    batch_scope: Literal["exactly-one-task-and-one-search-seed"] = (
        "exactly-one-task-and-one-search-seed"
    )
    study_id: NonEmptyStr
    preregistration_ref: ArtifactRef
    four_arm_design: ConfirmatoryFourArmDesign
    four_arm_design_ref: ArtifactRef
    pure_at_b_plan: PureAtBPlan
    pure_at_b_plan_ref: ArtifactRef
    model_spec: FrozenModelSpec
    model_spec_ref: ArtifactRef
    seed_harness: HarnessManifest
    seed_harness_ref: ArtifactRef
    solver_config_ref: ArtifactRef
    optimizer_config_ref: ArtifactRef
    candidate_parser_implementation_ref: ArtifactRef
    task_bindings: Annotated[
        tuple[ConfirmatoryTaskRuntimeBinding, ...],
        Field(min_length=1, max_length=1),
    ]
    search_seeds: Annotated[
        tuple[ConfirmatorySearchSeed, ...],
        Field(min_length=1, max_length=1),
    ]
    development_only: Literal[True] = True
    evidence_status: Literal["permanent-nonreportable-rehearsal"] = (
        "permanent-nonreportable-rehearsal"
    )
    live_model_calls_permitted: Literal[False] = False
    runtime_capability_attested: Literal[False] = False
    execution_attested: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    same_served_revision_claim: Literal[False] = False
    sealed_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    _condition_order: ClassVar[tuple[ConfirmatoryExecutionCondition, ...]] = _CONDITION_ORDER

    @field_validator("study_id", mode="before")
    @classmethod
    def _study_id_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("study_id must be exact and non-empty")
        return value

    @field_validator("task_bindings")
    @classmethod
    def _canonicalize_bindings(
        cls,
        values: tuple[ConfirmatoryTaskRuntimeBinding, ...],
    ) -> tuple[ConfirmatoryTaskRuntimeBinding, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.task_id))
        ids = tuple(item.task_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("confirmatory task runtime bindings must not repeat")
        return ordered

    @field_validator("search_seeds")
    @classmethod
    def _canonicalize_search_seeds(
        cls,
        values: tuple[ConfirmatorySearchSeed, ...],
    ) -> tuple[ConfirmatorySearchSeed, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.seed_id))
        ids = tuple(item.seed_id for item in ordered)
        fingerprints = tuple(item.fingerprint for item in ordered)
        raw_values = tuple(item.seed for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("confirmatory search seed IDs must not repeat")
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("confirmatory full search seeds must not repeat")
        if len(raw_values) != len(set(raw_values)):
            raise ValueError("confirmatory raw search seed values must not repeat")
        return ordered

    @model_validator(mode="after")
    def _close_design_and_runtime_bindings(self) -> Self:
        design_ref = _canonical_ref(
            self.four_arm_design,
            CONFIRMATORY_FOUR_ARM_DESIGN_MEDIA_TYPE,
        )
        if self.four_arm_design_ref != design_ref:
            raise ValueError("four-arm design ref differs from canonical design content")
        pure_ref = _canonical_ref(self.pure_at_b_plan, PURE_AT_B_PLAN_MEDIA_TYPE)
        if self.pure_at_b_plan_ref != pure_ref:
            raise ValueError("PURE@B plan ref differs from canonical plan content")
        if self.pure_at_b_plan.four_arm_design != self.four_arm_design:
            raise ValueError("PURE@B plan embeds a different four-arm design")
        model_ref = _canonical_ref(
            self.model_spec,
            CONFIRMATORY_FROZEN_MODEL_SPEC_MEDIA_TYPE,
        )
        if self.model_spec_ref != model_ref:
            raise ValueError("model spec ref differs from canonical FrozenModelSpec content")
        harness_ref = _canonical_ref(self.seed_harness, HARNESS_MANIFEST_MEDIA_TYPE)
        if self.seed_harness_ref != harness_ref:
            raise ValueError("seed harness ref differs from canonical manifest content")
        if self.seed_harness.parent is not None:
            raise ValueError("confirmatory seed harness must be a genesis manifest")

        full = self.four_arm_design.arm(RealTaskArm.FULL)
        topology = full.adaptive_topology
        if topology is None:  # pragma: no cover - ConfirmatoryFourArmDesign invariant
            raise ValueError("FULL adaptive topology is missing")
        commitments = topology.protocol_commitments
        if commitments.model_spec_fingerprint != self.model_spec.fingerprint:
            raise ValueError("FrozenModelSpec differs from the four-arm model commitment")
        if commitments.seed_harness_ref != self.seed_harness_ref:
            raise ValueError("seed harness differs from the four-arm commitment")
        committed_refs = (
            (
                "solver config",
                self.solver_config_ref,
                commitments.solver_config_fingerprint,
            ),
            (
                "optimizer config",
                self.optimizer_config_ref,
                commitments.optimizer_config_fingerprint,
            ),
            (
                "candidate parser",
                self.candidate_parser_implementation_ref,
                commitments.candidate_parser_fingerprint,
            ),
        )
        for label, ref, fingerprint in committed_refs:
            if ref.sha256 != fingerprint:
                raise ValueError(f"{label} ref differs from the four-arm commitment")
        if self.seed_harness.model_fingerprint != self.model_spec.model_fingerprint:
            raise ValueError("seed harness model fingerprint differs from FrozenModelSpec")
        if self.seed_harness.runtime_fingerprint != self.model_spec.runtime_fingerprint:
            raise ValueError("seed harness runtime fingerprint differs from FrozenModelSpec")

        split = commitments.task_split_manifest
        expected_tasks = {task.task_id: task for task in split.tasks}
        bound_tasks = {task.task_id: task for task in self.task_bindings}
        if set(bound_tasks) != set(expected_tasks):
            raise ValueError("task runtime bindings differ from the canonical task roster")
        for task_id, expected in expected_tasks.items():
            actual = bound_tasks[task_id]
            if actual.task_manifest_ref != expected.task_manifest_ref:
                raise ValueError("task runtime manifest ref differs from canonical roster")
            if actual.evaluation_units != expected.evaluation_units:
                raise ValueError("task runtime evaluation-unit roster differs from canonical split")

        bundle_expectations = {
            "grader_implementation_ref": commitments.grader_fingerprint,
            "query_dag_implementation_ref": commitments.query_dag_fingerprint,
            "retry_policy_implementation_ref": commitments.retry_policy_fingerprint,
        }
        for field_name, expected in bundle_expectations.items():
            actual = task_implementation_bundle_fingerprint(
                self.task_bindings,
                field_name,  # type: ignore[arg-type]  # narrowed by the literal mapping
            )
            if actual != expected:
                raise ValueError(f"{field_name} bundle differs from four-arm commitment")

        seed_schedule = confirmatory_search_seed_schedule_fingerprint(
            self.task_bindings,
            self.search_seeds,
        )
        if seed_schedule != commitments.seed_schedule_fingerprint:
            raise ValueError("search seed schedule differs from four-arm commitment")

        aggregation_by_task = {
            aggregation.task_id: aggregation for aggregation in self.pure_at_b_plan.aggregations
        }
        for task in self.task_bindings:
            aggregation = aggregation_by_task.get(task.task_id)
            if aggregation is None:  # pragma: no cover - PURE@B validates the task roster
                raise ValueError("PURE@B aggregation is missing for a bound task")
            if aggregation.implementation_fingerprint != task.aggregation_implementation_ref.sha256:
                raise ValueError("aggregation implementation ref differs from PURE@B")
            if aggregation.normalizer_fingerprint != task.aggregation_normalizer_ref.sha256:
                raise ValueError("aggregation normalizer ref differs from PURE@B")
            if aggregation.output_domain_fingerprint != task.aggregation_output_domain_ref.sha256:
                raise ValueError("aggregation output-domain ref differs from PURE@B")
        return self

    @property
    def model_role_bindings(self) -> tuple[ConfirmatoryModelRoleBinding, ...]:
        """Derive complete role coverage from exact ceilings and one model ref."""

        checked = type(self).model_validate(self._strict_python_content(), strict=True)
        full = checked.four_arm_design.arm(RealTaskArm.FULL)
        ceilings = full.adaptive_ceilings
        if ceilings is None:  # pragma: no cover - strict design invariant
            raise ValueError("FULL adaptive ceilings are missing")
        by_role = {item.role: item for item in ceilings.role_model_calls}
        return tuple(
            ConfirmatoryModelRoleBinding(
                role=role,
                enabled=by_role[role].max_calls > 0,
                model_spec_ref=(checked.model_spec_ref if by_role[role].max_calls > 0 else None),
            )
            for role in ModelMediatedRole
        )

    @property
    def execution_keys(self) -> tuple[ConfirmatoryExecutionKey, ...]:
        """Derive the complete 5 x task x seed matrix from validated plan content."""

        checked = type(self).model_validate(self._strict_python_content(), strict=True)
        keys: list[ConfirmatoryExecutionKey] = []
        for condition in self._condition_order:
            for task in checked.task_bindings:
                for search_seed in checked.search_seeds:
                    block_coordinate = {
                        "domain": "spiral-harness/confirmatory-rehearsal-matched-block/v1",
                        "study_id": checked.study_id,
                        "preregistration_ref": checked.preregistration_ref,
                        "four_arm_design_ref": checked.four_arm_design_ref,
                        "pure_at_b_plan_ref": checked.pure_at_b_plan_ref,
                        "model_spec_ref": checked.model_spec_ref,
                        "seed_harness_ref": checked.seed_harness_ref,
                        "solver_config_ref": checked.solver_config_ref,
                        "optimizer_config_ref": checked.optimizer_config_ref,
                        "candidate_parser_implementation_ref": (
                            checked.candidate_parser_implementation_ref
                        ),
                        "task_id": task.task_id,
                        "task_manifest_ref": task.task_manifest_ref,
                        "task_runtime_binding_fingerprint": task.fingerprint,
                        "search_seed_id": search_seed.seed_id,
                        "search_seed_fingerprint": search_seed.fingerprint,
                    }
                    block_fingerprint = canonical_sha256(block_coordinate)
                    keys.append(
                        ConfirmatoryExecutionKey(
                            study_id=checked.study_id,
                            preregistration_ref=checked.preregistration_ref,
                            four_arm_design_ref=checked.four_arm_design_ref,
                            pure_at_b_plan_ref=checked.pure_at_b_plan_ref,
                            model_spec_ref=checked.model_spec_ref,
                            seed_harness_ref=checked.seed_harness_ref,
                            solver_config_ref=checked.solver_config_ref,
                            optimizer_config_ref=checked.optimizer_config_ref,
                            candidate_parser_implementation_ref=(
                                checked.candidate_parser_implementation_ref
                            ),
                            condition=condition,
                            task_id=task.task_id,
                            task_manifest_ref=task.task_manifest_ref,
                            task_runtime_binding_fingerprint=task.fingerprint,
                            search_seed=search_seed,
                            matched_block_fingerprint=block_fingerprint,
                            key_fingerprint=canonical_sha256(
                                {
                                    "domain": ("spiral-harness/confirmatory-rehearsal-cell/v1"),
                                    "matched_block_fingerprint": block_fingerprint,
                                    "condition": condition,
                                }
                            ),
                        )
                    )
        expected = (
            len(self._condition_order) * len(checked.task_bindings) * len(checked.search_seeds)
        )
        if len(keys) != expected or len({item.key_fingerprint for item in keys}) != expected:
            raise ValueError("derived execution-key matrix is incomplete or duplicated")
        return tuple(keys)

    def verify_execution_key_membership(
        self,
        key: ConfirmatoryExecutionKey,
    ) -> ConfirmatoryExecutionKey:
        """Require an exact plan-derived cell key, not merely a self-consistent key."""

        checked = ConfirmatoryExecutionKey.model_validate(key, strict=True)
        expected = {item.key_fingerprint: item for item in self.execution_keys}
        if expected.get(checked.key_fingerprint) != checked:
            raise ValueError("execution key is not a member of this exact rehearsal plan")
        return checked

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        return _canonical_ref(self, CONFIRMATORY_REHEARSAL_PLAN_MEDIA_TYPE)


def make_confirmatory_rehearsal_plan(
    *,
    study_id: str,
    preregistration_ref: ArtifactRef,
    four_arm_design: ConfirmatoryFourArmDesign,
    pure_at_b_plan: PureAtBPlan,
    model_spec: FrozenModelSpec,
    seed_harness: HarnessManifest,
    solver_config_ref: ArtifactRef,
    optimizer_config_ref: ArtifactRef,
    candidate_parser_implementation_ref: ArtifactRef,
    task_bindings: tuple[ConfirmatoryTaskRuntimeBinding, ...],
    search_seeds: tuple[ConfirmatorySearchSeed, ...],
) -> ConfirmatoryRehearsalPlan:
    """Build the rehearsal without accepting keys or evidence claims."""

    design = ConfirmatoryFourArmDesign.model_validate(four_arm_design, strict=True)
    pure_at_b = PureAtBPlan.model_validate(pure_at_b_plan, strict=True)
    spec = FrozenModelSpec.model_validate(model_spec, strict=True)
    harness = HarnessManifest.model_validate(seed_harness, strict=True)
    bindings = tuple(
        ConfirmatoryTaskRuntimeBinding.model_validate(item, strict=True) for item in task_bindings
    )
    return ConfirmatoryRehearsalPlan(
        study_id=study_id,
        preregistration_ref=preregistration_ref,
        four_arm_design=design,
        four_arm_design_ref=_canonical_ref(
            design,
            CONFIRMATORY_FOUR_ARM_DESIGN_MEDIA_TYPE,
        ),
        pure_at_b_plan=pure_at_b,
        pure_at_b_plan_ref=_canonical_ref(pure_at_b, PURE_AT_B_PLAN_MEDIA_TYPE),
        model_spec=spec,
        model_spec_ref=_canonical_ref(spec, CONFIRMATORY_FROZEN_MODEL_SPEC_MEDIA_TYPE),
        seed_harness=harness,
        seed_harness_ref=_canonical_ref(harness, HARNESS_MANIFEST_MEDIA_TYPE),
        solver_config_ref=solver_config_ref,
        optimizer_config_ref=optimizer_config_ref,
        candidate_parser_implementation_ref=candidate_parser_implementation_ref,
        task_bindings=bindings,
        search_seeds=search_seeds,
    )


__all__ = [
    "CONFIRMATORY_FROZEN_MODEL_SPEC_MEDIA_TYPE",
    "CONFIRMATORY_REHEARSAL_PLAN_MEDIA_TYPE",
    "ConfirmatoryEvaluationUnitSeed",
    "ConfirmatoryExecutionCondition",
    "ConfirmatoryExecutionKey",
    "ConfirmatoryModelRoleBinding",
    "ConfirmatoryRehearsalPlan",
    "ConfirmatorySearchSeed",
    "ConfirmatoryTaskRuntimeBinding",
    "confirmatory_search_seed_schedule_fingerprint",
    "make_confirmatory_rehearsal_plan",
    "task_implementation_bundle_fingerprint",
]
