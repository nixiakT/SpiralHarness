"""Contracts for a prospective, question-only BFCL V4 development roster.

The roster is public development data.  These contracts cannot represent
hidden, sealed, official-suite, score-bearing, or execution-plan evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_identity import (
    BFCL_V4_RELEASE_VERSION,
    BFCL_V4_UPSTREAM_COMMIT,
    BFCL_V4_UPSTREAM_REPOSITORY,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_identities import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_QUESTION_BLOB_IDENTITIES,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_identities import (
    BFCL_V4_PILOT_ROW_IDENTITIES,
)
from spiral_harness.benchmark.bfcl_v4_question_source import (
    BFCL_V4_QUESTION_SOURCE_UPSTREAM_COMMIT,
    BfclV4QuestionSourceBinding,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_DOMAIN = "spiral-bfcl-v4-public-development-v2-selection/v1"
BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_DOMAIN = (
    "spiral-bfcl-v4-public-development-v2-v1-exclusion/v1"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_SALT = (
    "spiral-bfcl-v4-public-development-v2:2026-08-15:pre-v1-result-open"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_CANDIDATE_POOL_SHA256 = (
    "88cef8b2fe76409f35f735b85dc230af39c21d30af0b4457c6a2d8e1d283b064"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_ELIGIBLE_SINGLETONS_SHA256 = (
    "3069ad528ea31fcb5cae92b3e67945262a05ad50227d22b68e135275a25ab209"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_SHA256 = (
    "07a360ed6e32da5a94ee6d553aa31234f86cd37d024b411f351df4b8f0cc87c0"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROSTER_CONTENT_SHA256 = (
    "2d629219523faf7469f17b12192e8286cb5e0ea1988b424828636c465a703d1c"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT = (
    "3f6e410bd26466381090ced0c6144515281a9653e5d36429685ad898f168eae7"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_LEXICAL_AUDIT_SHA256 = (
    "c75fdfca954773007c9475818cf408facfe958081ed946cedb5d339880d21542"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_SOURCE_BINDINGS = tuple(
    BfclV4QuestionSourceBinding(
        git_path=f"berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_{category}.json",
        size=BFCL_V4_PUBLIC_DEVELOPMENT_V2_QUESTION_BLOB_IDENTITIES[category][0],
        sha256=BFCL_V4_PUBLIC_DEVELOPMENT_V2_QUESTION_BLOB_IDENTITIES[category][1],
    )
    for category in BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_TOTALS = {
    "simple_python": 7,
    "multiple": 6,
    "parallel": 6,
    "parallel_multiple": 6,
}
BFCL_V4_PUBLIC_DEVELOPMENT_V2_SPLIT_CATEGORY_QUOTAS = {
    "fit": {"simple_python": 2, "multiple": 1, "parallel": 1, "parallel_multiple": 1},
    "gate": {"simple_python": 1, "multiple": 1, "parallel": 1, "parallel_multiple": 1},
    "holdout": {
        "simple_python": 4,
        "multiple": 4,
        "parallel": 4,
        "parallel_multiple": 4,
    },
}

BfclV4PublicDevelopmentV2Category = Literal[
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
]


class BfclV4PublicDevelopmentV2Split(StrEnum):
    """Prospective public-development information partitions."""

    FIT = "fit"
    GATE = "gate"
    HOLDOUT = "holdout"


class BfclV4PublicDevelopmentV2AuditStatus(StrEnum):
    """Terminal status of the provider-free roster audit."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class BfclV4PublicDevelopmentV2FailureStage(StrEnum):
    """Safe rejection stage; no source exception text is retained."""

    CHECKOUT_LINEAGE = "checkout-lineage"
    QUESTION_SOURCE = "question-source"
    SOURCE_PARSE = "source-parse"
    V1_LINEAGE = "v1-lineage"
    FAMILY_GRAPH = "family-graph"
    DETERMINISTIC_SELECTION = "deterministic-selection"
    FROZEN_IDENTITY = "frozen-identity"
    FUNCTION_NAME_AUDIT = "function-name-audit"
    NEAR_DUPLICATE_AUDIT = "near-duplicate-audit"


def bfcl_v4_public_development_v2_v1_exclusion_sha256(
    task_ids: tuple[str, ...],
) -> str:
    """Bind the ordered v1 task roster excluded before the v2 draw."""

    if (
        not isinstance(task_ids, tuple)
        or not task_ids
        or len(set(task_ids)) != len(task_ids)
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
    ):
        raise ValueError("v1 exclusion requires an ordered tuple of unique task IDs")
    return canonical_sha256(
        {
            "domain": BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_DOMAIN,
            "task_ids": task_ids,
        }
    )


def bfcl_v4_public_development_v2_selection_token(
    category: str,
    task_id: str,
) -> str:
    """Hash one singleton family under the frozen public salt formula."""

    return hashlib.sha256(
        b"\0".join(
            item.encode("utf-8")
            for item in (
                BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_DOMAIN,
                BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_SALT,
                category,
                task_id,
            )
        )
    ).hexdigest()


class BfclV4PublicDevelopmentV2RosterEntry(ImmutableModel):
    """One answer-free row selected as a singleton source/semantic family."""

    schema_version: Literal["1"] = "1"
    ordinal: Annotated[int, Field(ge=0, lt=25, strict=True)]
    task_id: NonEmptyStr
    category: BfclV4PublicDevelopmentV2Category
    split: BfclV4PublicDevelopmentV2Split
    official_function_names: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    source_family_sha256: Sha256
    semantic_template_sha256: Sha256
    selection_token_sha256: Sha256
    family_component_task_count: Literal[1] = 1
    question_git_path: NonEmptyStr
    question_blob_size: Annotated[int, Field(gt=0, strict=True)]
    question_blob_sha256: Sha256
    row_size: Annotated[int, Field(gt=0, strict=True)]
    row_sha256: Sha256
    candidate_payload_sha256: Sha256
    possible_answer_identity_present: Literal[False] = False
    score_identity_present: Literal[False] = False

    @model_validator(mode="after")
    def _bind_frozen_question_identity(self) -> Self:
        if not self.task_id.startswith(f"{self.category}_"):
            raise ValueError("v2 task ID differs from its category")
        expected_path = (
            f"berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_{self.category}.json"
        )
        if self.question_git_path != expected_path:
            raise ValueError("v2 question path differs from its category")
        if (self.question_blob_size, self.question_blob_sha256) != (
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_QUESTION_BLOB_IDENTITIES[self.category]
        ):
            raise ValueError("v2 question blob identity changed")
        try:
            frozen = BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES[self.task_id]
        except KeyError as error:
            raise ValueError("v2 task is outside the frozen roster") from error
        observed = (
            self.row_size,
            self.row_sha256,
            self.candidate_payload_sha256,
            self.official_function_names,
            self.source_family_sha256,
            self.semantic_template_sha256,
            self.selection_token_sha256,
            self.split.value,
        )
        if observed != frozen:
            raise ValueError("v2 row, family, function, token, or split identity changed")
        if self.selection_token_sha256 != bfcl_v4_public_development_v2_selection_token(
            self.category,
            self.task_id,
        ):
            raise ValueError("v2 selection token differs from the salt-hash formula")
        if len(set(self.official_function_names)) != len(self.official_function_names):
            raise ValueError("v2 task repeats an official function name")
        return self


def bfcl_v4_public_development_v2_roster_content_sha256(
    roster: tuple[BfclV4PublicDevelopmentV2RosterEntry, ...],
) -> str:
    """Hash the exact prospective roster without an execution schedule."""

    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-development-v2-roster-content/v1",
            "upstream_commit": BFCL_V4_UPSTREAM_COMMIT,
            "selection_domain": BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_DOMAIN,
            "selection_salt": BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_SALT,
            "roster": roster,
        }
    )


class BfclV4PublicDevelopmentV2Manifest(ImmutableModel):
    """Frozen 5/4/16 public-development roster with no execution plan."""

    schema_version: Literal["1"] = "1"
    suite_id: Literal["bfcl-v4-public-development-v2@6ea57973"] = (
        "bfcl-v4-public-development-v2@6ea57973"
    )
    upstream_repository: Literal[BFCL_V4_UPSTREAM_REPOSITORY] = BFCL_V4_UPSTREAM_REPOSITORY
    upstream_commit: Literal[BFCL_V4_UPSTREAM_COMMIT] = BFCL_V4_UPSTREAM_COMMIT
    upstream_release: Literal[BFCL_V4_RELEASE_VERSION] = BFCL_V4_RELEASE_VERSION
    roster: Annotated[
        tuple[BfclV4PublicDevelopmentV2RosterEntry, ...],
        Field(min_length=25, max_length=25),
    ]
    roster_content_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROSTER_CONTENT_SHA256]
    v1_exclusion_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_SHA256
    )
    candidate_pool_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CANDIDATE_POOL_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CANDIDATE_POOL_SHA256
    )
    eligible_singletons_sha256: Literal[
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_ELIGIBLE_SINGLETONS_SHA256
    ] = BFCL_V4_PUBLIC_DEVELOPMENT_V2_ELIGIBLE_SINGLETONS_SHA256
    lexical_audit_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_LEXICAL_AUDIT_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_LEXICAL_AUDIT_SHA256
    )
    selection_domain: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_DOMAIN] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_DOMAIN
    )
    selection_salt: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_SALT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTION_SALT
    )
    selection_formula: Literal[
        "sha256(utf8(domain)||00||utf8(salt)||00||utf8(category)||00||utf8(task_id))"
    ] = "sha256(utf8(domain)||00||utf8(salt)||00||utf8(category)||00||utf8(task_id))"
    family_partition_unit: Literal["eligible-singleton-source-semantic-component"] = (
        "eligible-singleton-source-semantic-component"
    )
    approximate_near_duplicate_grouping_attested: Literal[False] = False
    independent_semantic_family_disjointness_attested: Literal[False] = False
    lexical_near_duplicate_audit_required: Literal[True] = True
    official_function_names_disjoint_across_splits_required: Literal[True] = True
    v1_family_disjoint_required: Literal[True] = True
    execution_seeds_frozen: Literal[False] = False
    call_plan_frozen: Literal[False] = False
    model_invoked: Literal[False] = False
    scores_read: Literal[False] = False
    runs_or_cas_read: Literal[False] = False
    possible_answers_read: Literal[False] = False
    prospective_public_development_only: Literal[True] = True
    questions_public: Literal[True] = True
    possible_answers_public: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    sealed_evidence: Literal[False] = False
    official_full_suite: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_roster(self) -> Self:
        if BFCL_V4_QUESTION_SOURCE_UPSTREAM_COMMIT != BFCL_V4_UPSTREAM_COMMIT:
            raise ValueError("v2 question-source and manifest commits differ")
        if tuple(BFCL_V4_PILOT_ROW_IDENTITIES) != BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS:
            raise ValueError("v2 v1-exclusion roster differs from question-only v1 identities")
        if self.v1_exclusion_sha256 != bfcl_v4_public_development_v2_v1_exclusion_sha256(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS
        ):
            raise ValueError("v2 v1-exclusion identity differs from its domain formula")
        if tuple(item.ordinal for item in self.roster) != tuple(range(25)):
            raise ValueError("v2 roster ordinals must be 0..24")
        if tuple(item.task_id for item in self.roster) != (
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS
        ):
            raise ValueError("v2 roster differs from the deterministic selected order")
        if set(BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS) & {item.task_id for item in self.roster}:
            raise ValueError("v2 roster contains a v1 task")
        expected_splits = (
            *((BfclV4PublicDevelopmentV2Split.FIT,) * 5),
            *((BfclV4PublicDevelopmentV2Split.GATE,) * 4),
            *((BfclV4PublicDevelopmentV2Split.HOLDOUT,) * 16),
        )
        if tuple(item.split for item in self.roster) != expected_splits:
            raise ValueError("v2 roster must retain the frozen 5/4/16 order")
        for split in BfclV4PublicDevelopmentV2Split:
            observed = Counter(item.category for item in self.roster if item.split is split)
            if observed != BFCL_V4_PUBLIC_DEVELOPMENT_V2_SPLIT_CATEGORY_QUOTAS[split.value]:
                raise ValueError("v2 split differs from the category-stratified quota")
        all_names = tuple(name for item in self.roster for name in item.official_function_names)
        if len(all_names) != 44 or len(set(all_names)) != 44:
            raise ValueError("v2 official function-name families are not globally disjoint")
        if (
            len({item.source_family_sha256 for item in self.roster}) != 25
            or len({item.semantic_template_sha256 for item in self.roster}) != 25
        ):
            raise ValueError("v2 selected source/semantic families are not singleton-disjoint")
        if self.roster_content_sha256 != (
            bfcl_v4_public_development_v2_roster_content_sha256(self.roster)
        ):
            raise ValueError("v2 roster content fingerprint changed")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicDevelopmentV2Task(ImmutableModel):
    """Candidate-visible question/schema bytes for one selected v2 task."""

    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    category: BfclV4PublicDevelopmentV2Category
    question_json: NonEmptyStr
    function_schemas_json: NonEmptyStr
    official_function_names: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    candidate_payload_sha256: Sha256
    runtime_git_object_read_attested: Literal[False] = False
    possible_answer_data_in_contract: Literal[False] = False
    answer_derived_identity_present: Literal[False] = False
    score_derived_identity_present: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_candidate_payload(self) -> Self:
        try:
            question = json.loads(self.question_json)
            functions = json.loads(self.function_schemas_json)
        except json.JSONDecodeError as error:
            raise ValueError("v2 candidate payload JSON is invalid") from error
        if canonical_json(question) != self.question_json or not isinstance(functions, list):
            raise ValueError("v2 candidate payload is not canonical question/schema JSON")
        if canonical_json(functions) != self.function_schemas_json:
            raise ValueError("v2 function schemas are not canonical JSON")
        names = tuple(item.get("name") for item in functions if isinstance(item, dict))
        try:
            frozen = BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES[self.task_id]
        except KeyError as error:
            raise ValueError("v2 loaded task is outside the frozen roster") from error
        if self.category != _category(self.task_id):
            raise ValueError("v2 loaded task category differs from its ID")
        if (
            names != self.official_function_names
            or names != frozen[3]
            or (len(names) != len(functions))
        ):
            raise ValueError("v2 loaded function names differ from the frozen row")
        payload = {"id": self.task_id, "question": question, "function": functions}
        if (
            canonical_sha256(payload) != self.candidate_payload_sha256
            or self.candidate_payload_sha256 != frozen[2]
        ):
            raise ValueError("v2 loaded candidate payload fingerprint changed")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicDevelopmentV2AuditReceipt(ImmutableModel):
    """Accepted or fail-closed rejected question-only roster audit."""

    schema_version: Literal["1"] = "1"
    status: BfclV4PublicDevelopmentV2AuditStatus
    failure_stage: BfclV4PublicDevelopmentV2FailureStage | None = None
    source_bindings: Annotated[tuple[BfclV4QuestionSourceBinding, ...], Field(max_length=4)] = ()
    candidate_pool_sha256: Sha256 | None = None
    eligible_singletons_sha256: Sha256 | None = None
    lexical_audit_sha256: Sha256 | None = None
    attempted_selected_task_ids: tuple[NonEmptyStr, ...] = ()
    manifest_fingerprint: Sha256 | None = None
    source_row_count: Annotated[int, Field(ge=0, le=1000, strict=True)] = 0
    eligible_singleton_count: Annotated[int, Field(ge=0, le=1000, strict=True)] = 0
    selected_task_count: Annotated[int, Field(ge=0, le=25, strict=True)] = 0
    lexical_comparison_count: Annotated[int, Field(ge=0, strict=True)] = 0
    lexical_near_duplicate_count: Annotated[int, Field(ge=0, strict=True)] = 0
    official_function_name_overlap_count: Annotated[int, Field(ge=0, strict=True)] = 0
    v1_task_overlap_count: Annotated[int, Field(ge=0, strict=True)] = 0
    v1_family_overlap_count: Annotated[int, Field(ge=0, strict=True)] = 0
    no_reselection_after_audit_failure: Literal[True] = True
    exception_text_persisted: Literal[False] = False
    possible_answers_read: Literal[False] = False
    scores_read: Literal[False] = False
    runs_or_cas_read: Literal[False] = False
    model_invoked: Literal[False] = False
    network_used: Literal[False] = False
    prospective_public_development_only: Literal[True] = True
    independent_semantic_family_disjointness_attested: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_terminal_status(self) -> Self:
        if self.status is BfclV4PublicDevelopmentV2AuditStatus.ACCEPTED:
            if (
                self.failure_stage is not None
                or self.source_bindings != BFCL_V4_PUBLIC_DEVELOPMENT_V2_SOURCE_BINDINGS
                or self.candidate_pool_sha256 != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CANDIDATE_POOL_SHA256
                or self.eligible_singletons_sha256
                != BFCL_V4_PUBLIC_DEVELOPMENT_V2_ELIGIBLE_SINGLETONS_SHA256
                or self.lexical_audit_sha256 != BFCL_V4_PUBLIC_DEVELOPMENT_V2_LEXICAL_AUDIT_SHA256
                or self.attempted_selected_task_ids
                != BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS
                or self.manifest_fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
                or self.source_row_count != 1000
                or self.eligible_singleton_count != 202
                or self.selected_task_count != 25
                or self.lexical_comparison_count != 539
                or any(
                    (
                        self.lexical_near_duplicate_count,
                        self.official_function_name_overlap_count,
                        self.v1_task_overlap_count,
                        self.v1_family_overlap_count,
                    )
                )
            ):
                raise ValueError("accepted v2 receipt lacks the exact closed audit")
        elif self.failure_stage is None or self.manifest_fingerprint is not None:
            raise ValueError("rejected v2 receipt requires a safe stage and no manifest")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4LoadedPublicDevelopmentV2(ImmutableModel):
    """Trusted control-plane bundle; never release this object to a candidate."""

    schema_version: Literal["1"] = "1"
    manifest: BfclV4PublicDevelopmentV2Manifest
    tasks: Annotated[tuple[BfclV4PublicDevelopmentV2Task, ...], Field(min_length=25, max_length=25)]
    audit_receipt: BfclV4PublicDevelopmentV2AuditReceipt
    trusted_control_plane_only: Literal[True] = True
    candidate_visible: Literal[False] = False
    execution_seeds_present: Literal[False] = False
    call_plan_present: Literal[False] = False
    model_invoked: Literal[False] = False
    possible_answers_read: Literal[False] = False
    scores_read: Literal[False] = False
    runs_or_cas_read: Literal[False] = False
    independent_semantic_family_disjointness_attested: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_loaded_roster(self) -> Self:
        if tuple((item.task_id, item.category) for item in self.tasks) != tuple(
            (item.task_id, item.category) for item in self.manifest.roster
        ):
            raise ValueError("loaded v2 tasks differ from the manifest roster")
        if (
            self.audit_receipt.status is not BfclV4PublicDevelopmentV2AuditStatus.ACCEPTED
            or self.audit_receipt.manifest_fingerprint != self.manifest.fingerprint
        ):
            raise ValueError("loaded v2 roster lacks its accepted audit receipt")
        return self


def _category(task_id: str) -> BfclV4PublicDevelopmentV2Category:
    for category in sorted(
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER,
        key=len,
        reverse=True,
    ):
        if task_id.startswith(f"{category}_"):
            return category  # type: ignore[return-value]
    raise ValueError("frozen v2 task has no supported category")


def _entry(ordinal: int, task_id: str) -> BfclV4PublicDevelopmentV2RosterEntry:
    category = _category(task_id)
    (
        row_size,
        row_sha256,
        candidate_payload_sha256,
        names,
        source_family_sha256,
        semantic_template_sha256,
        selection_token_sha256,
        split,
    ) = BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES[task_id]
    blob_size, blob_sha256 = BFCL_V4_PUBLIC_DEVELOPMENT_V2_QUESTION_BLOB_IDENTITIES[category]
    return BfclV4PublicDevelopmentV2RosterEntry(
        ordinal=ordinal,
        task_id=task_id,
        category=category,
        split=BfclV4PublicDevelopmentV2Split(split),
        official_function_names=names,
        source_family_sha256=source_family_sha256,
        semantic_template_sha256=semantic_template_sha256,
        selection_token_sha256=selection_token_sha256,
        question_git_path=(
            f"berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_{category}.json"
        ),
        question_blob_size=blob_size,
        question_blob_sha256=blob_sha256,
        row_size=row_size,
        row_sha256=row_sha256,
        candidate_payload_sha256=candidate_payload_sha256,
    )


BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROSTER = tuple(
    _entry(ordinal, task_id)
    for ordinal, task_id in enumerate(BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS)
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST = BfclV4PublicDevelopmentV2Manifest(
    roster=BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROSTER,
    roster_content_sha256=BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROSTER_CONTENT_SHA256,
)

__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_") or name.startswith("Bfcl") or name.startswith("bfcl_")
]
