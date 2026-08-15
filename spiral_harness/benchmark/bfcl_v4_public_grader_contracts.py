"""Fail-closed contracts for the public-development BFCL V4 grader plane.

The fifteen pilot rows and their answers are public upstream data.  These
contracts therefore never describe hidden, sealed, official-full-suite, or
reportable evidence.  They do, however, keep answer-derived identities and
exact checker diagnostics on a grader/auditor-only plane so the development
pilot can exercise the same information-flow discipline as a sealed study.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from spiral_harness.benchmark.bfcl_v4 import BFCL_V4_UPSTREAM_COMMIT
from spiral_harness.benchmark.bfcl_v4_fixture_contracts import (
    SOURCE_ISOLATED_EXPECTED_FILES,
    BfclV4EnvironmentEntry,
    BfclV4ExecutableBinding,
    BfclV4LocalFileBinding,
    BfclV4RuntimeCoordinates,
    BfclV4SourceFileBinding,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_bindings import (
    EXPECTED_ANSWER_BINDING_SHA256,
    EXPECTED_QUESTION_BINDING_SHA256,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PilotSplit,
)
from spiral_harness.core.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

PUBLIC_GRADER_PROTOCOL = "spiral-bfcl-v4-public-development-grader/v1"
PUBLIC_GRADER_FAILURE_PROTOCOL = "spiral-bfcl-v4-public-grader-failure/v1"
PUBLIC_GRADER_SCOPE = "public-development-partial-bfcl-pilot"

type PilotTaskId = Literal[
    "simple_python_0",
    "simple_python_211",
    "multiple_5",
    "parallel_0",
    "parallel_multiple_9",
    "multiple_10",
    "parallel_multiple_11",
    "simple_python_87",
    "simple_python_128",
    "multiple_7",
    "multiple_8",
    "parallel_3",
    "parallel_4",
    "parallel_multiple_5",
    "parallel_multiple_55",
]
type PilotCategory = Literal["simple_python", "multiple", "parallel", "parallel_multiple"]
type PilotArm = Literal["pure", "static", "score", "full", "pure-at-b"]
type PilotGradeRole = Literal[
    "baseline",
    "parent-fit",
    "candidate-fit",
    "gate-parent",
    "gate-candidate",
    "gate-revert",
    "gate-placebo",
    "holdout",
    "pure-at-b-selected",
]
type CoarseFailureClass = Literal[
    "none", "call-count", "function-or-arguments", "execution-failure"
]

_TASK_BY_ID = {item.task_id: item for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster}

WORKER_ENVIRONMENT_ITEMS = tuple(
    sorted(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }.items()
    )
)

_FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")


class BfclV4StrictGraderContract(ImmutableModel):
    """Revalidate instances created through Pydantic's unchecked APIs."""

    @classmethod
    def _raw(cls, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return {name: cls._raw(getattr(value, name)) for name in type(value).model_fields}
        if isinstance(value, Mapping):
            return {key: cls._raw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._raw(item) for item in value)
        if isinstance(value, list):
            return [cls._raw(item) for item in value]
        return value

    def _strict_content(self) -> dict[str, Any]:
        return {name: self._raw(getattr(self, name)) for name in type(self).model_fields}

    def _strict_revalidate(self) -> None:
        type(self).model_validate(self._strict_content(), strict=True)

    @model_serializer(mode="wrap")
    def _strict_serializer(self, handler: SerializerFunctionWrapHandler) -> Any:
        self._strict_revalidate()
        return handler(self)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        self._strict_revalidate()
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        self._strict_revalidate()
        return super().model_dump_json(**kwargs)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep
        content = self._strict_content()
        if update is not None:
            content.update(update)
        return type(self).model_validate(content, strict=True)


class BfclV4PublicDevelopmentContract(BfclV4StrictGraderContract):
    evidence_scope: Literal["public-development-partial-bfcl-pilot"] = PUBLIC_GRADER_SCOPE
    questions_public: Literal[True] = True
    possible_answers_public: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    sealed_evidence: Literal[False] = False
    official_full_suite: Literal[False] = False
    reportable_result: Literal[False] = False


def _reject_surrogates(value: Any) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON strings must not contain surrogate code points")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_surrogates(key)
            _reject_surrogates(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_surrogates(item)


def _validate_json_tree(value: Any, *, max_depth: int = 32, max_nodes: int = 10_000) -> None:
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise ValueError("JSON value exceeds the grader complexity limit")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite JSON numbers are forbidden")
        if isinstance(item, str):
            _reject_surrogates(item)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                _reject_surrogates(key)
                visit(child, depth + 1)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child, depth + 1)
        elif item is not None and not isinstance(item, (bool, int, float)):
            raise ValueError("unsupported JSON value")

    visit(value, 0)


def strict_json(content: str, label: str) -> Any:
    """Decode bounded JSON while rejecting duplicates, non-finite values, and surrogates."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate object keys")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError(f"{label} contains a non-finite number")

    if not isinstance(content, str):
        raise TypeError(f"{label} must be text")
    if len(content.encode("utf-8", errors="strict")) > 1_048_576:
        raise ValueError(f"{label} exceeds the byte limit")
    try:
        value = json.loads(
            content,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeEncodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    _validate_json_tree(value)
    return value


class BfclV4OfficialPredictionCall(BfclV4StrictGraderContract):
    """One candidate-submitted call after explicit wire-to-official recovery."""

    function_name: Annotated[NonEmptyStr, Field(max_length=256)]
    arguments_json: Annotated[NonEmptyStr, Field(max_length=1_048_576)]

    @field_validator("function_name")
    @classmethod
    def _official_name_is_safe(cls, value: str) -> str:
        _reject_surrogates(value)
        if value != value.strip() or _FUNCTION_NAME.fullmatch(value) is None:
            raise ValueError("function name is not an exact official-name token")
        return value

    @field_validator("arguments_json")
    @classmethod
    def _arguments_are_canonical_object(cls, value: str) -> str:
        arguments = strict_json(value, "prediction arguments")
        if not isinstance(arguments, dict):
            raise ValueError("prediction arguments must be a JSON object")
        if canonical_json(arguments) != value:
            raise ValueError("prediction arguments must use canonical JSON")
        return value


def prediction_content(prediction: BfclV4PublicPrediction) -> dict[str, Any]:
    return {
        "calls": [
            {"arguments_json": call.arguments_json, "function_name": call.function_name}
            for call in prediction.calls
        ],
        "task_id": prediction.task_id,
    }


class BfclV4PublicPrediction(BfclV4StrictGraderContract):
    """The complete candidate-plane submission surface: official calls only."""

    schema_version: Literal["1"] = "1"
    task_id: PilotTaskId
    calls: Annotated[tuple[BfclV4OfficialPredictionCall, ...], Field(max_length=64)] = ()

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(prediction_content(self))


class BfclV4GradingSlotBinding(BfclV4StrictGraderContract):
    """Runner-owned plan/slot coordinate bound into worker input and output."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    call_slot_reference_sha256: Sha256
    call_id: NonEmptyStr
    arm: PilotArm
    grade_role: PilotGradeRole
    intended_harness_variant: NonEmptyStr
    executed_harness_variant: NonEmptyStr
    fallback_used: bool = False
    task_id: PilotTaskId
    prediction_sha256: Sha256

    @field_validator("call_id", "intended_harness_variant", "executed_harness_variant")
    @classmethod
    def _coordinates_are_exact(cls, value: str) -> str:
        _reject_surrogates(value)
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("grading coordinates must be exact printable text")
        return value

    @model_validator(mode="after")
    def _role_matches_task_partition(self) -> Self:
        split = _TASK_BY_ID[self.task_id].split
        fit_roles = {"parent-fit", "candidate-fit"}
        gate_roles = {"gate-parent", "gate-candidate", "gate-revert", "gate-placebo"}
        holdout_roles = {"baseline", "holdout", "pure-at-b-selected"}
        if split == BfclV4PilotSplit.FIT and self.grade_role not in fit_roles:
            raise ValueError("FIT task uses a non-FIT grading role")
        if split == BfclV4PilotSplit.GATE and self.grade_role not in gate_roles:
            raise ValueError("GATE task uses a non-GATE grading role")
        if split == BfclV4PilotSplit.HOLDOUT and self.grade_role not in holdout_roles:
            raise ValueError("HOLDOUT task uses a non-HOLDOUT grading role")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4HoldoutUnlock(BfclV4StrictGraderContract):
    """Precondition that both adaptive-arm selections precede any holdout grade."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    score_selection_artifact_sha256: Sha256
    full_selection_artifact_sha256: Sha256
    both_selection_artifacts_frozen: Literal[True] = True
    selections_final_before_holdout_grading: Literal[True] = True
    holdout_can_continue_search: Literal[False] = False
    ordering_independently_attested: Literal[False] = False
    visibility: Literal["runner-grader-auditor-only"] = "runner-grader-auditor-only"
    candidate_visible: Literal[False] = False

    @model_validator(mode="after")
    def _selections_are_distinct(self) -> Self:
        if self.score_selection_artifact_sha256 == self.full_selection_artifact_sha256:
            raise ValueError("SCORE and FULL selection artifacts must be distinct")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def question_binding_content(binding: BfclV4PublicQuestionBinding) -> dict[str, Any]:
    return {
        "category": binding.category,
        "function_schemas_sha256": binding.function_schemas_sha256,
        "official_function_names": list(binding.official_function_names),
        "question_blob_sha256": binding.question_blob_sha256,
        "question_blob_size": binding.question_blob_size,
        "question_git_path": binding.question_git_path,
        "question_row_sha256": binding.question_row_sha256,
        "question_row_size": binding.question_row_size,
        "question_sha256": binding.question_sha256,
        "split": binding.split,
        "task_id": binding.task_id,
    }


class BfclV4PublicQuestionBinding(BfclV4StrictGraderContract):
    """Exact public question/schema bytes independently recomputed by the worker."""

    task_id: PilotTaskId
    category: PilotCategory
    split: BfclV4PilotSplit
    question_git_path: NonEmptyStr
    question_blob_size: Annotated[int, Field(gt=0, strict=True)]
    question_blob_sha256: Sha256
    question_row_size: Annotated[int, Field(gt=0, strict=True)]
    question_row_sha256: Sha256
    question_sha256: Sha256
    function_schemas_sha256: Sha256
    official_function_names: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _bind_frozen_public_row(self) -> Self:
        roster = _TASK_BY_ID[self.task_id]
        if (self.category, self.split, self.question_git_path) != (
            roster.category,
            roster.split,
            roster.question_git_path,
        ):
            raise ValueError("question binding differs from the frozen roster")
        if len(set(self.official_function_names)) != len(self.official_function_names):
            raise ValueError("question function names must be unique")
        if self.fingerprint != EXPECTED_QUESTION_BINDING_SHA256[self.task_id]:
            raise ValueError("question binding differs from the pinned public row")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(question_binding_content(self))


def answer_binding_content(binding: BfclV4GraderAnswerBinding) -> dict[str, Any]:
    return {
        "answer_blob_sha256": binding.answer_blob_sha256,
        "answer_blob_size": binding.answer_blob_size,
        "answer_git_path": binding.answer_git_path,
        "answer_row_sha256": binding.answer_row_sha256,
        "answer_row_size": binding.answer_row_size,
        "candidate_visible": False,
        "contains_answer_derived_identity": True,
        "ground_truth_sha256": binding.ground_truth_sha256,
        "task_id": binding.task_id,
        "visibility": "grader-auditor-only",
    }


class BfclV4GraderAnswerBinding(BfclV4StrictGraderContract):
    """Opaque answer identity emitted only by the isolated trusted worker."""

    task_id: PilotTaskId
    answer_git_path: NonEmptyStr
    answer_blob_size: Annotated[int, Field(gt=0, strict=True)]
    answer_blob_sha256: Sha256
    answer_row_size: Annotated[int, Field(gt=0, strict=True)]
    answer_row_sha256: Sha256
    ground_truth_sha256: Sha256
    visibility: Literal["grader-auditor-only"] = "grader-auditor-only"
    candidate_visible: Literal[False] = False
    contains_answer_derived_identity: Literal[True] = True

    @model_validator(mode="after")
    def _bind_frozen_public_answer(self) -> Self:
        roster = _TASK_BY_ID[self.task_id]
        expected_path = (
            "berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/"
            f"BFCL_v4_{roster.category}.json"
        )
        if self.answer_git_path != expected_path:
            raise ValueError("answer binding path differs from the frozen category")
        if self.fingerprint != EXPECTED_ANSWER_BINDING_SHA256[self.task_id]:
            raise ValueError("answer binding differs from the pinned public answer row")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(answer_binding_content(self))


class BfclV4PublicGraderReceipt(BfclV4PublicDevelopmentContract):
    """Complete trusted-plane receipt; never serialize into candidate context."""

    schema_version: Literal["1"] = "1"
    visibility: Literal["grader-auditor-only"] = "grader-auditor-only"
    candidate_visible: Literal[False] = False
    contains_answer_derived_identity: Literal[True] = True
    upstream_commit: Literal[BFCL_V4_UPSTREAM_COMMIT] = BFCL_V4_UPSTREAM_COMMIT
    prediction: BfclV4PublicPrediction
    prediction_sha256: Sha256
    slot: BfclV4GradingSlotBinding
    slot_binding_sha256: Sha256
    question_binding: BfclV4PublicQuestionBinding
    answer_binding: BfclV4GraderAnswerBinding
    holdout_unlock: BfclV4HoldoutUnlock | None = None
    holdout_unlock_sha256: Sha256 | None = None
    executed_sources: Annotated[tuple[BfclV4SourceFileBinding, ...], Field(min_length=5)]
    source_bundle_sha256: Sha256
    worker_source: BfclV4LocalFileBinding
    checkout_path: NonEmptyStr
    argv: Annotated[tuple[NonEmptyStr, ...], Field(min_length=8)]
    argv_sha256: Sha256
    environment: Annotated[tuple[BfclV4EnvironmentEntry, ...], Field(min_length=1)]
    environment_sha256: Sha256
    runtime: BfclV4RuntimeCoordinates
    python_executable: BfclV4ExecutableBinding
    git_executable: BfclV4ExecutableBinding
    worker_stdout_size: Annotated[int, Field(gt=0, strict=True)]
    worker_stdout_sha256: Sha256
    worker_stderr_observed_empty: Literal[True] = True
    returncode_observation: Literal[0] = 0
    subprocess_observation_independently_attested: Literal[False] = False
    runtime_execution_independently_attested: Literal[False] = False
    exact_upstream_ast_checker_executed: Literal[True] = True
    upstream_ast_checker_valid: bool
    checker_error_type: NonEmptyStr | None = None
    coarse_failure_class: CoarseFailureClass
    full_upstream_dependency_graph_loaded: Literal[False] = False
    provider_model_registry_stubbed: Literal[True] = True
    dependency_environment_attested: Literal[False] = False
    official_cli_executed: Literal[False] = False
    official_score_produced: Literal[False] = False
    model_invoked: Literal[False] = False
    network_isolation_attested: Literal[False] = False
    network_calls_requested: Literal[False] = False
    credential_environment_inherited: Literal[False] = False
    proxy_environment_inherited: Literal[False] = False

    @model_validator(mode="after")
    def _bind_receipt(self) -> Self:
        if self.prediction_sha256 != self.prediction.fingerprint:
            raise ValueError("receipt prediction fingerprint mismatch")
        if self.slot.prediction_sha256 != self.prediction_sha256:
            raise ValueError("slot points to another prediction")
        if self.slot.task_id != self.prediction.task_id:
            raise ValueError("slot points to another task")
        if self.slot_binding_sha256 != self.slot.fingerprint:
            raise ValueError("slot binding fingerprint mismatch")
        if self.question_binding.task_id != self.prediction.task_id:
            raise ValueError("question binding points to another task")
        if self.answer_binding.task_id != self.prediction.task_id:
            raise ValueError("answer binding points to another task")
        _require_exact_source_roster(self.executed_sources)
        if self.source_bundle_sha256 != source_bundle_sha256(self.executed_sources):
            raise ValueError("executed source bundle fingerprint mismatch")
        if self.worker_source != current_public_grader_worker_binding():
            raise ValueError("worker source differs from the current public grader worker")
        expected_argv = (
            self.python_executable.path,
            "-I",
            "-B",
            self.worker_source.path,
            "--checkout",
            self.checkout_path,
            "--git",
            self.git_executable.path,
        )
        if self.argv != expected_argv or self.argv_sha256 != canonical_sha256(self.argv):
            raise ValueError("grader argv is not bound to its execution coordinates")
        environment_items = tuple((item.name, item.value) for item in self.environment)
        if environment_items != WORKER_ENVIRONMENT_ITEMS:
            raise ValueError("worker environment differs from the exact allowlist")
        expected_environment = canonical_sha256(
            {
                "entries": self.environment,
                "git": self.git_executable,
                "platform": self.runtime,
                "python": self.python_executable,
            }
        )
        if self.environment_sha256 != expected_environment:
            raise ValueError("grader environment fingerprint mismatch")
        split = self.question_binding.split
        if split == BfclV4PilotSplit.HOLDOUT:
            if self.holdout_unlock is None:
                raise ValueError("HOLDOUT grading requires both frozen selection artifacts")
            if self.holdout_unlock.plan_fingerprint != self.slot.plan_fingerprint:
                raise ValueError("HOLDOUT unlock belongs to another plan")
            if self.holdout_unlock_sha256 != self.holdout_unlock.fingerprint:
                raise ValueError("HOLDOUT unlock fingerprint mismatch")
        elif self.holdout_unlock is not None or self.holdout_unlock_sha256 is not None:
            raise ValueError("non-HOLDOUT grade must not carry a HOLDOUT unlock")
        if self.upstream_ast_checker_valid != (self.checker_error_type is None):
            raise ValueError("checker validity and exact error type disagree")
        expected_coarse = coarse_failure_class(
            self.upstream_ast_checker_valid,
            self.checker_error_type,
        )
        if self.coarse_failure_class != expected_coarse:
            raise ValueError("coarse failure class differs from the exact checker result")
        stdout = canonical_json_bytes(worker_success_payload(self)) + b"\n"
        if len(stdout) != self.worker_stdout_size:
            raise ValueError("worker stdout size differs from reconstructable payload")
        if sha256_bytes(stdout) != self.worker_stdout_sha256:
            raise ValueError("worker stdout hash differs from reconstructable payload")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def checked[ModelT: BfclV4StrictGraderContract](value: ModelT, model: type[ModelT]) -> ModelT:
    if not isinstance(value, model):
        raise TypeError(f"expected {model.__name__}")
    return model.model_validate(value._strict_content(), strict=True)


def source_bundle_sha256(bindings: tuple[BfclV4SourceFileBinding, ...]) -> str:
    return canonical_sha256(
        [{"path": item.git_path, "sha256": item.sha256, "size": item.size} for item in bindings]
    )


def _require_exact_source_roster(
    bindings: tuple[BfclV4SourceFileBinding, ...],
) -> None:
    expected_paths = tuple(sorted(SOURCE_ISOLATED_EXPECTED_FILES))
    if tuple(item.git_path for item in bindings) != expected_paths:
        raise ValueError("grader source roster differs from the pinned ordered roster")
    for item in bindings:
        expected_size, expected_sha256 = SOURCE_ISOLATED_EXPECTED_FILES[item.git_path]
        if (item.size, item.sha256) != (expected_size, expected_sha256):
            raise ValueError(f"grader source size or hash differs: {item.git_path}")


def current_public_grader_worker_binding() -> BfclV4LocalFileBinding:
    worker = Path(__file__).with_name("_bfcl_v4_public_grader_worker.py").resolve(strict=True)
    content = worker.read_bytes()
    return BfclV4LocalFileBinding(
        path=str(worker),
        size=len(content),
        sha256=sha256_bytes(content),
    )


def runtime_coordinates() -> BfclV4RuntimeCoordinates:
    cache_tag = sys.implementation.cache_tag
    if not cache_tag:
        raise RuntimeError("Python runtime cache tag is unavailable")
    return BfclV4RuntimeCoordinates(
        implementation=platform.python_implementation().casefold(),
        python_version=platform.python_version(),
        cache_tag=cache_tag,
        system=platform.system().casefold(),
        machine=platform.machine().casefold(),
    )


def worker_environment() -> dict[str, str]:
    return dict(WORKER_ENVIRONMENT_ITEMS)


def coarse_failure_class(valid: bool, error_type: str | None) -> CoarseFailureClass:
    if valid:
        if error_type is not None:
            raise ValueError("valid checker result retained an error type")
        return "none"
    if not isinstance(error_type, str) or not error_type:
        raise ValueError("invalid checker result omitted its error type")
    if "wrong_count" in error_type:
        return "call-count"
    return "function-or-arguments"


def worker_success_payload(receipt: BfclV4PublicGraderReceipt) -> dict[str, Any]:
    return {
        "answer_binding": answer_binding_content(receipt.answer_binding),
        "coarse_failure_class": receipt.coarse_failure_class,
        "credential_environment_inherited": False,
        "error_type": receipt.checker_error_type,
        "exact_upstream_ast_checker_executed": True,
        "full_upstream_dependency_graph_loaded": False,
        "git_executable_sha256": receipt.git_executable.sha256_observation,
        "holdout_unlock_sha256": receipt.holdout_unlock_sha256,
        "model_invoked": False,
        "network_calls_requested": False,
        "network_isolation_attested": False,
        "official_cli_executed": False,
        "official_score_produced": False,
        "prediction_sha256": receipt.prediction_sha256,
        "protocol": PUBLIC_GRADER_PROTOCOL,
        "proxy_environment_inherited": False,
        "question_binding": question_binding_content(receipt.question_binding),
        "slot_binding_sha256": receipt.slot_binding_sha256,
        "source_bundle_sha256": receipt.source_bundle_sha256,
        "task_id": receipt.prediction.task_id,
        "upstream_ast_checker_valid": receipt.upstream_ast_checker_valid,
    }


__all__ = [
    "EXPECTED_ANSWER_BINDING_SHA256",
    "EXPECTED_QUESTION_BINDING_SHA256",
    "PUBLIC_GRADER_FAILURE_PROTOCOL",
    "PUBLIC_GRADER_PROTOCOL",
    "BfclV4GraderAnswerBinding",
    "BfclV4GradingSlotBinding",
    "BfclV4HoldoutUnlock",
    "BfclV4OfficialPredictionCall",
    "BfclV4PublicDevelopmentContract",
    "BfclV4PublicGraderReceipt",
    "BfclV4PublicPrediction",
    "BfclV4PublicQuestionBinding",
    "BfclV4StrictGraderContract",
    "answer_binding_content",
    "checked",
    "coarse_failure_class",
    "current_public_grader_worker_binding",
    "prediction_content",
    "question_binding_content",
    "runtime_coordinates",
    "source_bundle_sha256",
    "strict_json",
    "worker_environment",
    "worker_success_payload",
]
