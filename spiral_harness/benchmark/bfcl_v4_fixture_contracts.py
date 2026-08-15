"""Typed, permanently non-reportable contracts for the BFCL V4 fixture bridge."""

from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from spiral_harness.benchmark._bfcl_v4_fixture_worker import (
    ANSWER_BLOB_SHA256,
    ANSWER_PATH,
    ANSWER_ROW_SHA256,
    CATEGORY,
    QUESTION_BLOB_SHA256,
    QUESTION_PATH,
    TASK_ID,
    TASK_ROW_SHA256,
    UPSTREAM_COMMIT,
)
from spiral_harness.core.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    module_source_sha256,
    sha256_bytes,
)
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

UPSTREAM_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
UPSTREAM_RELEASE = "2026.3.23"
SUITE_ID = "bfcl-v4@6ea57973"
FIXTURE_PROTOCOL = "spiral-bfcl-v4-source-isolated-fixture/v1"
FIXTURE_SCOPE = "public-partial-source-isolated-fixture-rehearsal-only"
FUNCTION_SCHEMAS_SHA256 = "78b19dceb27af200400b076e1f8de46ebf13bf086163b9aae12b34be7020a62e"
QUESTION_SHA256 = "416b5e059443ba5b56f89998bd61ebdc6426828325b2487b25df82fa03e0f2b6"
GROUND_TRUTH_SHA256 = "48730d0bdcf0db0216edfce9f256f18f96c1b0312ffd6754f00702a2d2cd1501"
QUESTION_BLOB_SIZE = 283_274
ANSWER_BLOB_SIZE = 63_627
TASK_ROW_SIZE = 613
ANSWER_ROW_SIZE = 127

OPENAI_DECODER_PATH = (
    "berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/openai_completion.py"
)
OPENAI_DECODER_SHA256 = "f0bc8734a3173df79e52fa6d757ac6b2585f7582e2f19ac7cddf5170269dd3f7"
EVAL_RUNNER_PATH = "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/eval_runner.py"
EVAL_RUNNER_SHA256 = "b1033684908819ccb312d4d0e2c563359d69247412f00206013b2292b0e3ce81"
CLI_PATH = "berkeley-function-call-leaderboard/bfcl_eval/__main__.py"
CLI_SHA256 = "c70b49a7541abeaa399d193dd0d94d535f83ce415bfbbf7aa217c2370d14fcb6"
PYPROJECT_PATH = "berkeley-function-call-leaderboard/pyproject.toml"
PYPROJECT_SHA256 = "21233eb88f9d5ba195ff935766e9e13106043dfb0ef6a1a5637b6a4246abdfc4"

SOURCE_ISOLATED_EXPECTED_FILES = {
    "berkeley-function-call-leaderboard/bfcl_eval/constants/enums.py": (
        896,
        "2182becfa2a1d071ee1db30db593b4758c6bf866aa12d2d4b8daf09175ea518a",
    ),
    "berkeley-function-call-leaderboard/bfcl_eval/constants/type_mappings.py": (
        1_813,
        "1702fb67afbe2c492608e58e2b7d02e46381f50166b47f3c952f76e34c7cd3bd",
    ),
    (
        "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/"
        "type_convertor/java_type_converter.py"
    ): (15_716, "2fd4f4b0443b3dd974a1723bb4e45c086d7b352631062da7807ad1ad40706604"),
    (
        "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/"
        "type_convertor/js_type_converter.py"
    ): (11_706, "a114e9ff75c025cb52787ac33d6c2fbaa390905c6125a2b3c6afebab232bb5e4"),
    "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py": (
        25_032,
        "2aae7a68461a8f76c0be3894c8901b66b56967a1989d3ab066051e3fb97f1538",
    ),
}
FULL_CLI_EXPECTED_FILES = {
    CLI_PATH: (11_285, CLI_SHA256),
    EVAL_RUNNER_PATH: (32_756, EVAL_RUNNER_SHA256),
    OPENAI_DECODER_PATH: (11_291, OPENAI_DECODER_SHA256),
    PYPROJECT_PATH: (2_013, PYPROJECT_SHA256),
}
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

FULL_CLI_ARGV_TEMPLATE = (
    "{python_executable}",
    "-I",
    "-B",
    "-m",
    "bfcl_eval",
    "evaluate",
    "--model",
    "{registered_fc_decoder}",
    "--test-category",
    CATEGORY,
    "--result-dir",
    "{bfcl_project_relative_result_dir}",
    "--score-dir",
    "{bfcl_project_relative_score_dir}",
    "--partial-eval",
)
FULL_CLI_ENVIRONMENT_REQUIREMENTS = (
    "install-bfcl-eval-from-exact-git-commit-6ea57973",
    "freeze-complete-transitive-dependency-lock-and-python-runtime",
    "use-empty-isolated-dotenv-because-upstream-evaluate-loads-dotenv-override-true",
    "deny-outbound-network-during-evaluation",
    "use-a-registered-fc-handler-with-openai-style-result-row-decoding",
    "record-result-and-score-directories-by-content-not-mutable-path-alone",
)


class _StrictFixtureModel(ImmutableModel):
    """Revalidate even instances created through Pydantic's unchecked APIs."""

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


class _PublicPartialNonReportable(_StrictFixtureModel):
    evidence_scope: Literal["public-partial-source-isolated-fixture-rehearsal-only"] = FIXTURE_SCOPE
    partial_evaluation: Literal[True] = True
    questions_public: Literal[True] = True
    possible_answers_public: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    sealed_evidence: Literal[False] = False
    reportable_result: Literal[False] = False
    official_score_produced: Literal[False] = False
    model_invoked: Literal[False] = False


class BfclV4SourceFileBinding(_StrictFixtureModel):
    git_path: NonEmptyStr
    size: Annotated[int, Field(gt=0, strict=True)]
    sha256: Sha256

    @field_validator("git_path")
    @classmethod
    def _path_is_canonical(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != value:
            raise ValueError("source git path must be canonical and repository-relative")
        return value


class BfclV4PublicAstFixture(_PublicPartialNonReportable):
    """Candidate-visible task bytes and schemas with no answer-derived identity."""

    schema_version: Literal["1"] = "1"
    suite_id: Literal["bfcl-v4@6ea57973"] = SUITE_ID
    upstream_repository: Literal[UPSTREAM_REPOSITORY] = UPSTREAM_REPOSITORY
    upstream_commit: Literal[UPSTREAM_COMMIT] = UPSTREAM_COMMIT
    upstream_release: Literal[UPSTREAM_RELEASE] = UPSTREAM_RELEASE
    task_id: Literal["simple_python_0"] = TASK_ID
    category: Literal["simple_python"] = CATEGORY
    question_git_path: Literal[QUESTION_PATH] = QUESTION_PATH
    question_blob_size: Literal[QUESTION_BLOB_SIZE] = QUESTION_BLOB_SIZE
    question_blob_sha256: Literal[QUESTION_BLOB_SHA256] = QUESTION_BLOB_SHA256
    task_row_size: Literal[TASK_ROW_SIZE] = TASK_ROW_SIZE
    task_row_sha256: Literal[TASK_ROW_SHA256] = TASK_ROW_SHA256
    question_sha256: Literal[QUESTION_SHA256] = QUESTION_SHA256
    function_schemas_json: NonEmptyStr
    function_schemas_sha256: Literal[FUNCTION_SCHEMAS_SHA256] = FUNCTION_SCHEMAS_SHA256
    function_names: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    score_free_adapter: Literal[True] = True

    @model_validator(mode="after")
    def _bind_schemas(self) -> Self:
        schemas = strict_json(self.function_schemas_json, "function schemas")
        if not isinstance(schemas, list) or not schemas:
            raise ValueError("function schemas must be a non-empty JSON array")
        if canonical_json(schemas) != self.function_schemas_json:
            raise ValueError("function schemas must use canonical JSON")
        if canonical_sha256(schemas) != self.function_schemas_sha256:
            raise ValueError("function schema fingerprint mismatch")
        names = tuple(item.get("name") for item in schemas if isinstance(item, dict))
        if len(names) != len(schemas) or any(not isinstance(name, str) for name in names):
            raise ValueError("every function schema must have a string name")
        if names != self.function_names or len(set(names)) != len(names):
            raise ValueError("function-name roster differs from the schema bytes")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4GraderOnlyAnswerBinding(_StrictFixtureModel):
    """Trusted-worker answer identity, forbidden from candidate serialization."""

    schema_version: Literal["1"] = "1"
    upstream_commit: Literal[UPSTREAM_COMMIT] = UPSTREAM_COMMIT
    task_id: Literal["simple_python_0"] = TASK_ID
    answer_git_path: Literal[ANSWER_PATH] = ANSWER_PATH
    answer_blob_size: Literal[ANSWER_BLOB_SIZE] = ANSWER_BLOB_SIZE
    answer_blob_sha256: Literal[ANSWER_BLOB_SHA256] = ANSWER_BLOB_SHA256
    answer_row_size: Literal[ANSWER_ROW_SIZE] = ANSWER_ROW_SIZE
    answer_row_sha256: Literal[ANSWER_ROW_SHA256] = ANSWER_ROW_SHA256
    ground_truth_sha256: Literal[GROUND_TRUTH_SHA256] = GROUND_TRUTH_SHA256
    visibility: Literal["grader-auditor-only"] = "grader-auditor-only"
    candidate_visible: Literal[False] = False
    contains_answer_derived_identity: Literal[True] = True
    reportable_result: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4NativeToolCall(_StrictFixtureModel):
    function_name: NonEmptyStr
    arguments_json: NonEmptyStr

    @field_validator("function_name", mode="before")
    @classmethod
    def _function_name_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (
            value != value.strip() or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("function name must be exact and contain no control characters")
        return value

    @field_validator("arguments_json")
    @classmethod
    def _arguments_are_a_canonical_object(cls, value: str) -> str:
        arguments = strict_json(value, "tool-call arguments")
        if not isinstance(arguments, dict):
            raise ValueError("tool-call arguments must be a JSON object")
        if canonical_json(arguments) != value:
            raise ValueError("tool-call arguments must use canonical JSON")
        return value


class BfclV4OfficialResultExport(_PublicPartialNonReportable):
    schema_version: Literal["1"] = "1"
    fixture: BfclV4PublicAstFixture
    calls: Annotated[tuple[BfclV4NativeToolCall, ...], Field(min_length=1, max_length=1)]
    native_response_sha256: Sha256
    serializer_id: Literal["spiral-bfcl-openai-fc-jsonl/v1"] = "spiral-bfcl-openai-fc-jsonl/v1"
    serializer_source_sha256: Sha256
    upstream_decoder_git_path: Literal[OPENAI_DECODER_PATH] = OPENAI_DECODER_PATH
    upstream_decoder_sha256: Literal[OPENAI_DECODER_SHA256] = OPENAI_DECODER_SHA256
    response_row_jsonl: Annotated[bytes, Field(min_length=1)]
    response_row_size: Annotated[int, Field(gt=0, strict=True)]
    response_row_sha256: Sha256

    @model_validator(mode="after")
    def _bind_export(self) -> Self:
        if self.native_response_sha256 != canonical_sha256(self.calls):
            raise ValueError("native response fingerprint mismatch")
        if self.serializer_source_sha256 != fixture_contract_source_sha256():
            raise ValueError("serializer source fingerprint mismatch")
        expected_row = serialize_official_result_row(self.fixture, self.calls)
        if self.response_row_jsonl != expected_row:
            raise ValueError("response row differs from the bound native tool calls")
        if len(expected_row) != self.response_row_size:
            raise ValueError("response row size mismatch")
        if sha256_bytes(expected_row) != self.response_row_sha256:
            raise ValueError("response row fingerprint mismatch")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4ExecutableBinding(_StrictFixtureModel):
    path: NonEmptyStr
    size_observation: Annotated[int, Field(gt=0, strict=True)]
    sha256_observation: Sha256
    version_observation: NonEmptyStr
    independently_attested: Literal[False] = False


class BfclV4LocalFileBinding(_StrictFixtureModel):
    path: NonEmptyStr
    size: Annotated[int, Field(gt=0, strict=True)]
    sha256: Sha256


class BfclV4RuntimeCoordinates(_StrictFixtureModel):
    implementation: NonEmptyStr
    python_version: NonEmptyStr
    cache_tag: NonEmptyStr
    system: NonEmptyStr
    machine: NonEmptyStr


class BfclV4EnvironmentEntry(_StrictFixtureModel):
    name: NonEmptyStr
    value: NonEmptyStr


class BfclV4SourceIsolatedReceipt(_PublicPartialNonReportable):
    schema_version: Literal["1"] = "1"
    visibility: Literal["grader-auditor-only"] = "grader-auditor-only"
    candidate_visible: Literal[False] = False
    contains_answer_derived_identity: Literal[True] = True
    fixture_export_observation_sha256: Sha256
    worker_recomputed_task_binding_sha256: Sha256
    grader_answer_binding: BfclV4GraderOnlyAnswerBinding
    upstream_commit: Literal[UPSTREAM_COMMIT] = UPSTREAM_COMMIT
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
    response_row_observation_sha256: Sha256
    worker_stdout_size: Annotated[int, Field(gt=0, strict=True)]
    worker_stdout_sha256: Sha256
    worker_stderr_observed_empty: Literal[True] = True
    returncode_observation: Literal[0] = 0
    subprocess_observation_independently_attested: Literal[False] = False
    upstream_ast_checker_valid: bool
    checker_error_type: NonEmptyStr | None = None
    exact_upstream_ast_checker_executed: Literal[True] = True
    full_upstream_dependency_graph_loaded: Literal[False] = False
    provider_model_registry_stubbed: Literal[True] = True
    dependency_environment_attested: Literal[False] = False
    official_cli_executed: Literal[False] = False
    network_isolation_attested: Literal[False] = False
    network_calls_requested: Literal[False] = False
    credential_environment_inherited: Literal[False] = False
    proxy_environment_inherited: Literal[False] = False

    @model_validator(mode="after")
    def _bind_receipt(self) -> Self:
        _require_exact_source_roster(
            self.executed_sources,
            SOURCE_ISOLATED_EXPECTED_FILES,
            "source-isolated grader",
        )
        if self.source_bundle_sha256 != source_bundle_sha256(self.executed_sources):
            raise ValueError("executed source bundle fingerprint mismatch")
        if self.worker_source != current_worker_source_binding():
            raise ValueError("worker source differs from the current pinned worker")
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
        if self.argv != expected_argv:
            raise ValueError("grader argv is not bound to executable, worker, and checkout paths")
        if self.argv_sha256 != canonical_sha256(self.argv):
            raise ValueError("grader argv fingerprint mismatch")
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
        if self.upstream_ast_checker_valid != (self.checker_error_type is None):
            raise ValueError("checker validity and normalized error type disagree")
        stdout = canonical_json_bytes(worker_success_payload(self)) + b"\n"
        if len(stdout) != self.worker_stdout_size:
            raise ValueError("worker stdout size differs from reconstructable payload")
        if sha256_bytes(stdout) != self.worker_stdout_sha256:
            raise ValueError("worker stdout hash differs from reconstructable payload")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4CandidateSafeFeedback(_StrictFixtureModel):
    """Minimal projection that cannot carry trusted-plane identifiers."""

    schema_version: Literal["1"] = "1"
    information_scope: Literal["binary-public-fixture-feedback"] = "binary-public-fixture-feedback"
    task_id: Literal["simple_python_0"] = TASK_ID
    response_row_reference_sha256: Sha256
    accepted: bool
    failure_class: Literal["none", "ast-mismatch"]
    candidate_visible: Literal[True] = True
    partial_evaluation: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_feedback(self) -> Self:
        if self.accepted != (self.failure_class == "none"):
            raise ValueError("candidate feedback acceptance and failure class disagree")
        return self


class BfclV4FullCliInvocationContract(_PublicPartialNonReportable):
    """Frozen official-CLI template, permanently recording that it was not run."""

    schema_version: Literal["1"] = "1"
    upstream_repository: Literal[UPSTREAM_REPOSITORY] = UPSTREAM_REPOSITORY
    upstream_commit: Literal[UPSTREAM_COMMIT] = UPSTREAM_COMMIT
    upstream_release: Literal[UPSTREAM_RELEASE] = UPSTREAM_RELEASE
    source_files: Annotated[tuple[BfclV4SourceFileBinding, ...], Field(min_length=4)]
    source_bundle_sha256: Sha256
    package_pyproject_sha256: Literal[PYPROJECT_SHA256] = PYPROJECT_SHA256
    install_spec: Literal[
        "git+https://github.com/ShishirPatil/gorilla.git@"
        "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8#"
        "subdirectory=berkeley-function-call-leaderboard"
    ] = (
        "git+https://github.com/ShishirPatil/gorilla.git@"
        "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8#"
        "subdirectory=berkeley-function-call-leaderboard"
    )
    argv_template: tuple[NonEmptyStr, ...] = FULL_CLI_ARGV_TEMPLATE
    argv_template_sha256: Sha256
    environment_requirements: tuple[NonEmptyStr, ...] = FULL_CLI_ENVIRONMENT_REQUIREMENTS
    environment_contract_sha256: Sha256
    official_cli_executed: Literal[False] = False
    invocation_ready: Literal[False] = False
    dependency_environment_attested: Literal[False] = False
    registered_decoder_attested: Literal[False] = False
    provider_free_handler_construction_attested: Literal[False] = False
    blocker: Literal[
        "upstream-evaluate-imports-all-provider-handlers-and-instantiates-a-registered-handler"
    ] = "upstream-evaluate-imports-all-provider-handlers-and-instantiates-a-registered-handler"

    @model_validator(mode="after")
    def _bind_cli_contract(self) -> Self:
        _require_exact_source_roster(
            self.source_files,
            FULL_CLI_EXPECTED_FILES,
            "full BFCL CLI",
        )
        if self.source_bundle_sha256 != source_bundle_sha256(self.source_files):
            raise ValueError("full CLI source bundle fingerprint mismatch")
        pyproject = next(item for item in self.source_files if item.git_path == PYPROJECT_PATH)
        if pyproject.sha256 != self.package_pyproject_sha256:
            raise ValueError("full CLI pyproject binding mismatch")
        if self.argv_template != FULL_CLI_ARGV_TEMPLATE:
            raise ValueError("full CLI argv template differs from the pinned contract")
        if self.argv_template_sha256 != canonical_sha256(self.argv_template):
            raise ValueError("full CLI argv template fingerprint mismatch")
        if self.environment_requirements != FULL_CLI_ENVIRONMENT_REQUIREMENTS:
            raise ValueError("full CLI environment requirements changed")
        if self.environment_contract_sha256 != canonical_sha256(self.environment_requirements):
            raise ValueError("full CLI environment contract fingerprint mismatch")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def strict_json(content: str, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate object keys")
            result[key] = value
        return result

    try:
        return json.loads(content, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def checked[ModelT: _StrictFixtureModel](value: ModelT, model: type[ModelT]) -> ModelT:
    if not isinstance(value, model):
        raise TypeError(f"expected {model.__name__}")
    return model.model_validate(value._strict_content(), strict=True)


def fixture_contract_source_sha256() -> str:
    return module_source_sha256(sys.modules[__name__])


def serialize_official_result_row(
    fixture: BfclV4PublicAstFixture,
    calls: tuple[BfclV4NativeToolCall, ...],
) -> bytes:
    return (
        canonical_json_bytes(
            {
                "id": fixture.task_id,
                "result": [{call.function_name: call.arguments_json} for call in calls],
            }
        )
        + b"\n"
    )


def source_bundle_sha256(bindings: tuple[BfclV4SourceFileBinding, ...]) -> str:
    records = [
        {"path": item.git_path, "sha256": item.sha256, "size": item.size} for item in bindings
    ]
    return canonical_sha256(records)


def _require_exact_source_roster(
    bindings: tuple[BfclV4SourceFileBinding, ...],
    expected: Mapping[str, tuple[int, str]],
    label: str,
) -> None:
    expected_paths = tuple(sorted(expected))
    actual_paths = tuple(item.git_path for item in bindings)
    if actual_paths != expected_paths:
        raise ValueError(f"{label} source roster differs from the pinned ordered roster")
    for item in bindings:
        expected_size, expected_sha256 = expected[item.git_path]
        if (item.size, item.sha256) != (expected_size, expected_sha256):
            raise ValueError(f"{label} source size or hash differs: {item.git_path}")


def current_worker_source_binding() -> BfclV4LocalFileBinding:
    worker = Path(__file__).with_name("_bfcl_v4_fixture_worker.py").resolve(strict=True)
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
        machine=platform.machine().casefold(),
        python_version=platform.python_version(),
        cache_tag=cache_tag,
        system=platform.system().casefold(),
    )


def worker_environment() -> dict[str, str]:
    return dict(WORKER_ENVIRONMENT_ITEMS)


def worker_success_payload(receipt: BfclV4SourceIsolatedReceipt) -> dict[str, Any]:
    return {
        "credential_environment_inherited": False,
        "error_type": receipt.checker_error_type,
        "exact_upstream_ast_checker_executed": True,
        "full_upstream_dependency_graph_loaded": False,
        "git_executable_sha256": receipt.git_executable.sha256_observation,
        "grader_answer_binding_sha256": receipt.grader_answer_binding.fingerprint,
        "model_invoked": False,
        "network_calls_requested": False,
        "network_isolation_attested": False,
        "official_cli_executed": False,
        "official_score_produced": False,
        "protocol": FIXTURE_PROTOCOL,
        "proxy_environment_inherited": False,
        "response_row_sha256": receipt.response_row_observation_sha256,
        "source_bundle_sha256": receipt.source_bundle_sha256,
        "task_binding_sha256": receipt.worker_recomputed_task_binding_sha256,
        "upstream_ast_checker_valid": receipt.upstream_ast_checker_valid,
    }


__all__ = [
    "BfclV4CandidateSafeFeedback",
    "BfclV4EnvironmentEntry",
    "BfclV4ExecutableBinding",
    "BfclV4FullCliInvocationContract",
    "BfclV4GraderOnlyAnswerBinding",
    "BfclV4LocalFileBinding",
    "BfclV4NativeToolCall",
    "BfclV4OfficialResultExport",
    "BfclV4PublicAstFixture",
    "BfclV4RuntimeCoordinates",
    "BfclV4SourceFileBinding",
    "BfclV4SourceIsolatedReceipt",
]
