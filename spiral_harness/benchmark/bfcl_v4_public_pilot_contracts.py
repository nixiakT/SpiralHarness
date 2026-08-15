"""Prospective contracts for the public/development BFCL V4 pilot.

These values deliberately cannot represent hidden, sealed, official-full-suite,
or reportable evidence.  BFCL publishes both questions and possible answers.
The candidate-facing loader in the sibling runtime module nevertheless keeps
possible answers outside the selection process so that the pilot can exercise
the intended information boundary.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.benchmark.bfcl_v4 import (
    BFCL_V4_RELEASE_VERSION,
    BFCL_V4_UPSTREAM_COMMIT,
    BFCL_V4_UPSTREAM_REPOSITORY,
)
from spiral_harness.benchmark.bfcl_v4_fixture_contracts import BfclV4SourceFileBinding
from spiral_harness.benchmark.bfcl_v4_public_pilot_identities import (
    BFCL_V4_PILOT_ADAPTER_SOURCE_IDENTITIES,
    BFCL_V4_PILOT_QUESTION_BLOB_IDENTITIES,
    BFCL_V4_PILOT_ROW_IDENTITIES,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_PILOT_EXTERNAL_ROSTER_COMMITMENT = (
    "3556bca5d47d625fa3ba1cc086d6ac5c71b11a5327b16abb1d142c8e81ea1fee"
)
BFCL_V4_PILOT_EXTERNAL_SEED_COMMITMENT = (
    "4d210dfc8bc99e795cd8e58ed913a876180c4cf28f4310b2af68650e4e924042"
)
BFCL_V4_PILOT_OUTER_SEED_U64 = 2_026_081_501
BFCL_V4_PILOT_ROSTER_CONTENT_SHA256 = (
    "23677fd135db0f9beee3b8ea3853b48c1c4e924a1250897d97b53c64d7af4e37"
)

_WIRE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class BfclV4PilotSplit(StrEnum):
    """Frozen public-development information partitions."""

    FIT = "fit"
    GATE = "gate"
    HOLDOUT = "holdout"


class BfclV4PilotRosterEntry(ImmutableModel):
    """One preselected public BFCL row; no answer-derived field is allowed."""

    schema_version: Literal["1"] = "1"
    ordinal: Annotated[int, Field(ge=0, lt=15, strict=True)]
    task_id: NonEmptyStr
    category: Literal["simple_python", "multiple", "parallel", "parallel_multiple"]
    split: BfclV4PilotSplit
    semantic_family: NonEmptyStr
    question_git_path: NonEmptyStr
    question_blob_size: Annotated[int, Field(gt=0, strict=True)]
    question_blob_sha256: Sha256
    row_size: Annotated[int, Field(gt=0, strict=True)]
    row_sha256: Sha256
    candidate_payload_sha256: Sha256

    @model_validator(mode="after")
    def _bind_category_and_path(self) -> Self:
        if not self.task_id.startswith(f"{self.category}_"):
            raise ValueError("pilot task ID does not belong to its category")
        expected = f"berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_{self.category}.json"
        if self.question_git_path != expected:
            raise ValueError("pilot question path differs from its category")
        if (self.question_blob_size, self.question_blob_sha256) != (
            BFCL_V4_PILOT_QUESTION_BLOB_IDENTITIES[self.category]
        ):
            raise ValueError("pilot question blob identity differs from its category")
        if (self.row_size, self.row_sha256, self.candidate_payload_sha256) != (
            BFCL_V4_PILOT_ROW_IDENTITIES[self.task_id]
        ):
            raise ValueError("pilot row identity differs from its pinned task")
        return self


def bfcl_v4_pilot_roster_content_sha256(
    roster: tuple[BfclV4PilotRosterEntry, ...],
) -> str:
    """Hash the typed roster under this implementation's documented formula."""

    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-pilot-roster-content/v1",
            "upstream_commit": BFCL_V4_UPSTREAM_COMMIT,
            "roster": roster,
        }
    )


class BfclV4PublicPilotManifest(ImmutableModel):
    """Frozen 5/2/8 roster and explicit public-development claim boundary."""

    schema_version: Literal["1"] = "1"
    suite_id: Literal["bfcl-v4@6ea57973"] = "bfcl-v4@6ea57973"
    upstream_repository: Literal[BFCL_V4_UPSTREAM_REPOSITORY] = BFCL_V4_UPSTREAM_REPOSITORY
    upstream_commit: Literal[BFCL_V4_UPSTREAM_COMMIT] = BFCL_V4_UPSTREAM_COMMIT
    upstream_release: Literal[BFCL_V4_RELEASE_VERSION] = BFCL_V4_RELEASE_VERSION
    roster: Annotated[tuple[BfclV4PilotRosterEntry, ...], Field(min_length=15, max_length=15)]
    roster_content_sha256: Literal[BFCL_V4_PILOT_ROSTER_CONTENT_SHA256]
    external_roster_commitment_sha256: Literal[BFCL_V4_PILOT_EXTERNAL_ROSTER_COMMITMENT] = (
        BFCL_V4_PILOT_EXTERNAL_ROSTER_COMMITMENT
    )
    external_commitment_derivation_attested: Literal[False] = False
    external_commitment_equals_content_fingerprint_attested: Literal[False] = False
    evidence_scope: Literal["public-development-partial-bfcl-pilot"] = (
        "public-development-partial-bfcl-pilot"
    )
    questions_public: Literal[True] = True
    possible_answers_public: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    sealed_evidence: Literal[False] = False
    official_full_suite: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_roster(self) -> Self:
        if tuple(item.ordinal for item in self.roster) != tuple(range(15)):
            raise ValueError("pilot roster ordinals must be the frozen sequence 0..14")
        if len({item.task_id for item in self.roster}) != 15:
            raise ValueError("pilot task IDs must not repeat")
        if len({item.semantic_family for item in self.roster}) != 15:
            raise ValueError("pilot semantic families must not repeat")
        expected_splits = (
            *((BfclV4PilotSplit.FIT,) * 5),
            *((BfclV4PilotSplit.GATE,) * 2),
            *((BfclV4PilotSplit.HOLDOUT,) * 8),
        )
        if tuple(item.split for item in self.roster) != expected_splits:
            raise ValueError("pilot roster must retain the frozen FIT/GATE/HOLDOUT order")
        if self.roster_content_sha256 != bfcl_v4_pilot_roster_content_sha256(self.roster):
            raise ValueError("pilot roster content fingerprint differs from typed content")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicPilotTask(ImmutableModel):
    """Candidate-visible question/schema bytes from one pinned public row."""

    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    category: Literal["simple_python", "multiple", "parallel", "parallel_multiple"]
    split: BfclV4PilotSplit
    semantic_family: NonEmptyStr
    question_git_path: NonEmptyStr
    question_blob_size: Annotated[int, Field(gt=0, strict=True)]
    question_blob_sha256: Sha256
    row_size: Annotated[int, Field(gt=0, strict=True)]
    row_sha256: Sha256
    candidate_payload_sha256: Sha256
    question_json: NonEmptyStr
    function_schemas_json: NonEmptyStr
    official_function_names: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    runtime_git_object_read_attested: Literal[False] = False
    possible_answer_data_in_contract: Literal[False] = False
    answer_derived_identity_present: Literal[False] = False
    questions_public: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_candidate_visible_payload(self) -> Self:
        try:
            question = json.loads(self.question_json)
            functions = json.loads(self.function_schemas_json)
        except json.JSONDecodeError as error:
            raise ValueError("pilot task JSON is invalid") from error
        if canonical_json(question) != self.question_json:
            raise ValueError("pilot question JSON is not canonical")
        if (
            not isinstance(functions, list)
            or canonical_json(functions) != self.function_schemas_json
        ):
            raise ValueError("pilot function schemas are not a canonical JSON list")
        names = tuple(item.get("name") for item in functions if isinstance(item, dict))
        if names != self.official_function_names or len(names) != len(functions):
            raise ValueError("pilot function-name binding differs from schemas")
        if len(names) != len(set(names)):
            raise ValueError("function names must not repeat inside a pilot task")
        try:
            roster_item = next(
                item for item in BFCL_V4_PUBLIC_PILOT_ROSTER if item.task_id == self.task_id
            )
        except StopIteration as error:
            raise ValueError("pilot task is absent from the frozen roster") from error
        coordinates = (
            self.category,
            self.split,
            self.semantic_family,
            self.question_git_path,
            self.question_blob_size,
            self.question_blob_sha256,
            self.row_size,
            self.row_sha256,
        )
        expected_coordinates = (
            roster_item.category,
            roster_item.split,
            roster_item.semantic_family,
            roster_item.question_git_path,
            roster_item.question_blob_size,
            roster_item.question_blob_sha256,
            roster_item.row_size,
            roster_item.row_sha256,
        )
        if coordinates != expected_coordinates:
            raise ValueError("pilot task source coordinates differ from the frozen roster")
        payload_sha256 = canonical_sha256(
            {"id": self.task_id, "question": question, "function": functions}
        )
        if (
            self.candidate_payload_sha256 != roster_item.candidate_payload_sha256
            or payload_sha256 != self.candidate_payload_sha256
        ):
            raise ValueError("pilot candidate payload fingerprint differs from the pinned row")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4LoadedPublicPilot(ImmutableModel):
    """All candidate-visible rows; still only public/development evidence."""

    schema_version: Literal["1"] = "1"
    manifest: BfclV4PublicPilotManifest
    tasks: Annotated[tuple[BfclV4PublicPilotTask, ...], Field(min_length=15, max_length=15)]
    runtime_git_object_reads_attested: Literal[False] = False
    possible_answer_data_in_contract: Literal[False] = False
    model_invoked: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_loaded_rows(self) -> Self:
        roster_projection = tuple(
            (item.task_id, item.category, item.split, item.semantic_family)
            for item in self.manifest.roster
        )
        task_projection = tuple(
            (item.task_id, item.category, item.split, item.semantic_family) for item in self.tasks
        )
        if task_projection != roster_projection:
            raise ValueError("loaded pilot tasks differ from the frozen roster")
        names = tuple(name for task in self.tasks for name in task.official_function_names)
        if len(names) != 26 or len(set(names)) != 26:
            raise ValueError("public pilot must retain 26 globally unique official function names")
        return self


class BfclV4ToolNameBinding(ImmutableModel):
    """Explicit reversible official-name to provider wire-name binding."""

    official_name: NonEmptyStr
    wire_name: NonEmptyStr

    @model_validator(mode="after")
    def _bind_substitution(self) -> Self:
        if self.wire_name != self.official_name.replace(".", "_"):
            raise ValueError("wire name must be the pinned dot-to-underscore substitution")
        if _WIRE_NAME.fullmatch(self.wire_name) is None:
            raise ValueError("wire function name violates the OpenAI completions name grammar")
        return self


class BfclV4OpenAiToolAdapterResult(ImmutableModel):
    """Provider-ready schemas plus a collision-checked reverse mapping."""

    schema_version: Literal["1"] = "1"
    upstream_commit: Literal[BFCL_V4_UPSTREAM_COMMIT] = BFCL_V4_UPSTREAM_COMMIT
    model_style: Literal["openai-completions"] = "openai-completions"
    tools_json: NonEmptyStr
    name_bindings: Annotated[tuple[BfclV4ToolNameBinding, ...], Field(min_length=1)]
    upstream_converter_coordinates_bound: Literal[True] = True
    source_verification_receipt_present: Literal[False] = False
    upstream_handler_equivalence_attested: Literal[False] = False
    runtime_execution_attested: Literal[False] = False

    @model_validator(mode="after")
    def _close_adapter(self) -> Self:
        try:
            tools = json.loads(self.tools_json)
        except json.JSONDecodeError as error:
            raise ValueError("adapted tools JSON is invalid") from error
        if not isinstance(tools, list) or canonical_json(tools) != self.tools_json:
            raise ValueError("adapted tools must be a canonical JSON list")
        if len(tools) != len(self.name_bindings):
            raise ValueError("adapted tools and name bindings differ in length")
        official = tuple(item.official_name for item in self.name_bindings)
        wire = tuple(item.wire_name for item in self.name_bindings)
        if len(set(official)) != len(official) or len(set(wire)) != len(wire):
            raise ValueError("adapter name bindings must be bijective")
        tool_names = tuple(item.get("function", {}).get("name") for item in tools)
        if tool_names != wire:
            raise ValueError("adapted tool names differ from explicit wire bindings")
        return self

    @property
    def tools(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self.tools_json))

    def official_to_wire(self, name: str) -> str:
        matches = tuple(item.wire_name for item in self.name_bindings if item.official_name == name)
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def wire_to_official(self, name: str) -> str:
        matches = tuple(item.official_name for item in self.name_bindings if item.wire_name == name)
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]


class BfclV4AdapterSourceVerification(ImmutableModel):
    """Exact upstream-source binding, not an execution/equivalence receipt."""

    schema_version: Literal["1"] = "1"
    upstream_commit: Literal[BFCL_V4_UPSTREAM_COMMIT] = BFCL_V4_UPSTREAM_COMMIT
    model_style: Literal["openai-completions"] = "openai-completions"
    sources: Annotated[tuple[BfclV4SourceFileBinding, ...], Field(min_length=2, max_length=2)]
    source_bundle_sha256: Sha256
    source_snapshot_bound: Literal[True] = True
    runtime_execution_attested: Literal[False] = False
    upstream_handler_equivalence_attested: Literal[False] = False

    @model_validator(mode="after")
    def _bind_sources(self) -> Self:
        actual_sources = {item.git_path: (item.size, item.sha256) for item in self.sources}
        if actual_sources != BFCL_V4_PILOT_ADAPTER_SOURCE_IDENTITIES:
            raise ValueError("adapter sources differ from the exact pinned converter sources")
        expected = canonical_sha256(
            tuple(
                {"git_path": item.git_path, "sha256": item.sha256, "size": item.size}
                for item in self.sources
            )
        )
        if self.source_bundle_sha256 != expected:
            raise ValueError("adapter source bundle fingerprint differs")
        return self


class BfclV4WireToolCall(ImmutableModel):
    """One provider-wire call retained for target-free normalization."""

    wire_name: NonEmptyStr
    arguments_json: NonEmptyStr

    @field_validator("wire_name", "arguments_json", mode="before")
    @classmethod
    def _preserve_exact_text(cls, value: object) -> object:
        if isinstance(value, str) and value != value.strip():
            raise ValueError("wire call text must not have surrounding whitespace")
        return value


class BfclV4PureAtBSample(ImmutableModel):
    """One frozen-order sample; ``None`` means provider-level abstention."""

    sample_id: NonEmptyStr
    calls: tuple[BfclV4WireToolCall, ...] | None


class BfclV4PureAtBSelection(ImmutableModel):
    """Auditable plurality selection with no answer/grader input surface."""

    schema_version: Literal["1"] = "1"
    selected_sample_id: NonEmptyStr
    selected_frozen_index: Annotated[int, Field(ge=0, strict=True)]
    selected_calls: tuple[BfclV4WireToolCall, ...] | None
    normalized_output_json: str | None
    plurality_count: Annotated[int, Field(ge=0, strict=True)]
    tie: bool
    all_abstained: bool
    tie_breaker: Literal["first-sample-in-frozen-order"] = "first-sample-in-frozen-order"
    failed_sample_policy: Literal["abstain-and-all-abstain-select-first"] = (
        "abstain-and-all-abstain-select-first"
    )
    calls_sorted_canonically_with_multiplicity: Literal[True] = True
    grader_feedback_used: Literal[False] = False
    possible_answers_used: Literal[False] = False
    adaptive: Literal[False] = False


def _entry(
    ordinal: int,
    task_id: str,
    category: Literal["simple_python", "multiple", "parallel", "parallel_multiple"],
    split: BfclV4PilotSplit,
    semantic_family: str,
) -> BfclV4PilotRosterEntry:
    question_blob_size, question_blob_sha256 = BFCL_V4_PILOT_QUESTION_BLOB_IDENTITIES[category]
    row_size, row_sha256, candidate_payload_sha256 = BFCL_V4_PILOT_ROW_IDENTITIES[task_id]
    return BfclV4PilotRosterEntry(
        ordinal=ordinal,
        task_id=task_id,
        category=category,
        split=split,
        semantic_family=semantic_family,
        question_git_path=(
            f"berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_{category}.json"
        ),
        question_blob_size=question_blob_size,
        question_blob_sha256=question_blob_sha256,
        row_size=row_size,
        row_sha256=row_sha256,
        candidate_payload_sha256=candidate_payload_sha256,
    )


BFCL_V4_PUBLIC_PILOT_ROSTER = (
    _entry(0, "simple_python_0", "simple_python", BfclV4PilotSplit.FIT, "triangle-area"),
    _entry(1, "simple_python_211", "simple_python", BfclV4PilotSplit.FIT, "email-send"),
    _entry(2, "multiple_5", "multiple", BfclV4PilotSplit.FIT, "historical-weather"),
    _entry(3, "parallel_0", "parallel", BfclV4PilotSplit.FIT, "music-playback"),
    _entry(4, "parallel_multiple_9", "parallel_multiple", BfclV4PilotSplit.FIT, "travel-booking"),
    _entry(5, "multiple_10", "multiple", BfclV4PilotSplit.GATE, "database-column-privacy"),
    _entry(
        6, "parallel_multiple_11", "parallel_multiple", BfclV4PilotSplit.GATE, "electromagnetism"
    ),
    _entry(7, "simple_python_87", "simple_python", BfclV4PilotSplit.HOLDOUT, "array-sort"),
    _entry(8, "simple_python_128", "simple_python", BfclV4PilotSplit.HOLDOUT, "dividend-per-share"),
    _entry(9, "multiple_7", "multiple", BfclV4PilotSplit.HOLDOUT, "wildlife-ecology"),
    _entry(10, "multiple_8", "multiple", BfclV4PilotSplit.HOLDOUT, "real-estate-search"),
    _entry(11, "parallel_3", "parallel", BfclV4PilotSplit.HOLDOUT, "protein-models"),
    _entry(12, "parallel_4", "parallel", BfclV4PilotSplit.HOLDOUT, "body-mass-index"),
    _entry(13, "parallel_multiple_5", "parallel_multiple", BfclV4PilotSplit.HOLDOUT, "gcd-lcm"),
    _entry(
        14, "parallel_multiple_55", "parallel_multiple", BfclV4PilotSplit.HOLDOUT, "healthy-recipe"
    ),
)

BFCL_V4_PUBLIC_PILOT_MANIFEST = BfclV4PublicPilotManifest(
    roster=BFCL_V4_PUBLIC_PILOT_ROSTER,
    roster_content_sha256=bfcl_v4_pilot_roster_content_sha256(BFCL_V4_PUBLIC_PILOT_ROSTER),
)


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
