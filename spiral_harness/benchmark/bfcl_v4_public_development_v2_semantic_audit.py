"""Answer-free semantic-family audit structure for the BFCL v2 roster."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from itertools import combinations
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_identity import BFCL_V4_UPSTREAM_COMMIT
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SOURCE_BINDINGS,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_identities import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_identities import (
    BFCL_V4_PILOT_ROW_IDENTITIES,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_SEMANTIC_AUDIT_ITEM_DOMAIN = "spiral-bfcl-v4-semantic-audit-item/v1"
BFCL_V4_SEMANTIC_AUDIT_ORDER_DOMAIN = "spiral-bfcl-v4-semantic-audit-order/v1"
BFCL_V4_SEMANTIC_AUDIT_ORDER_SALT = "spiral-bfcl-v4-public-development-v2:2026-08-15:semantic-audit"
BFCL_V4_SEMANTIC_AUDIT_SOURCE_BUNDLE_SHA256 = canonical_sha256(
    {
        "domain": "spiral-bfcl-v4-semantic-audit-question-sources/v1",
        "upstream_commit": BFCL_V4_UPSTREAM_COMMIT,
        "source_bindings": BFCL_V4_PUBLIC_DEVELOPMENT_V2_SOURCE_BINDINGS,
    }
)


class BfclV4SemanticReviewerKind(StrEnum):
    """Declared origin of one independently generated review artifact."""

    HUMAN = "human"
    MODEL_ASSISTED = "model-assisted"


class BfclV4SemanticAuditFailureStage(StrEnum):
    """Fail-closed semantic-audit verification stage."""

    EVIDENCE_BINDING = "evidence-binding"
    REVIEW_INDEPENDENCE = "review-independence"
    DISAGREEMENT_ARBITRATION = "disagreement-arbitration"
    FAMILY_BOUNDARY = "family-boundary"


class BfclV4SemanticAuditError(RuntimeError):
    """Typed rejection with no annotation or lower-level exception text."""

    def __init__(self, failure_stage: BfclV4SemanticAuditFailureStage) -> None:
        self.failure_stage = failure_stage
        super().__init__(f"BFCL v2 semantic audit rejected at {failure_stage.value}")


def _category(task_id: str) -> str:
    for category in sorted(BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER, key=len, reverse=True):
        if task_id.startswith(f"{category}_"):
            return category
    raise ValueError("semantic-audit task has no supported category")


def _frozen_row_identity(task_id: str) -> tuple[int, str, str]:
    if task_id in BFCL_V4_PILOT_ROW_IDENTITIES:
        return BFCL_V4_PILOT_ROW_IDENTITIES[task_id]
    row = BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES[task_id]
    return row[0], row[1], row[2]


def _item_id(task_id: str, candidate_payload_sha256: str) -> str:
    return hashlib.sha256(
        b"\0".join(
            (
                BFCL_V4_SEMANTIC_AUDIT_ITEM_DOMAIN.encode(),
                task_id.encode(),
                candidate_payload_sha256.encode(),
            )
        )
    ).hexdigest()


def _order_token(task_id: str) -> str:
    return hashlib.sha256(
        b"\0".join(
            (
                BFCL_V4_SEMANTIC_AUDIT_ORDER_DOMAIN.encode(),
                BFCL_V4_SEMANTIC_AUDIT_ORDER_SALT.encode(),
                task_id.encode(),
            )
        )
    ).hexdigest()


class BfclV4SemanticAuditItem(ImmutableModel):
    """Trusted question-row binding; not a reviewer-facing projection."""

    schema_version: Literal["1"] = "1"
    audit_item_id: Sha256
    task_id: NonEmptyStr
    question_git_path: NonEmptyStr
    row_size: Annotated[int, Field(gt=0, strict=True)]
    row_sha256: Sha256
    candidate_payload_sha256: Sha256
    audit_order_token_sha256: Sha256
    explicit_partition_label_field_present: Literal[False] = False
    answer_identity_present: Literal[False] = False
    score_identity_present: Literal[False] = False

    @model_validator(mode="after")
    def _bind_question_row(self) -> Self:
        row_size, row_sha256, candidate_sha256 = _frozen_row_identity(self.task_id)
        category = _category(self.task_id)
        if (
            self.question_git_path
            != f"berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_{category}.json"
            or (self.row_size, self.row_sha256, self.candidate_payload_sha256)
            != (row_size, row_sha256, candidate_sha256)
            or self.audit_item_id != _item_id(self.task_id, candidate_sha256)
            or self.audit_order_token_sha256 != _order_token(self.task_id)
        ):
            raise ValueError("semantic-audit item differs from its frozen question row")
        return self


class BfclV4SemanticAuditUniverse(ImmutableModel):
    """Trusted binding universe from which a review packet must be derived."""

    schema_version: Literal["1"] = "1"
    manifest_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    )
    upstream_commit: Literal[BFCL_V4_UPSTREAM_COMMIT] = BFCL_V4_UPSTREAM_COMMIT
    question_source_bundle_sha256: Literal[BFCL_V4_SEMANTIC_AUDIT_SOURCE_BUNDLE_SHA256] = (
        BFCL_V4_SEMANTIC_AUDIT_SOURCE_BUNDLE_SHA256
    )
    items: Annotated[tuple[BfclV4SemanticAuditItem, ...], Field(min_length=40, max_length=40)]
    population: Literal["v1-and-v2-selected-question-rows"] = "v1-and-v2-selected-question-rows"
    explicit_partition_label_field_present: Literal[False] = False
    task_ids_and_paths_present: Literal[True] = True
    partition_inference_prevented: Literal[False] = False
    trusted_control_plane_only: Literal[True] = True
    reviewer_facing: Literal[False] = False
    reviewer_facing_projection_implemented: Literal[False] = False
    answers_present: Literal[False] = False
    scores_present: Literal[False] = False
    holdout_scores_present: Literal[False] = False
    model_or_harness_results_present: Literal[False] = False
    candidate_visible: Literal[False] = False

    @model_validator(mode="after")
    def _close_population(self) -> Self:
        expected_ids = BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS + (
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS
        )
        expected_order = tuple(
            sorted(expected_ids, key=lambda task_id: (_order_token(task_id), task_id))
        )
        if tuple(item.task_id for item in self.items) != expected_order:
            raise ValueError("semantic-audit universe differs from its frozen binding order")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4SemanticReviewer(ImmutableModel):
    """Declared reviewer provenance; declarations are not identity authentication."""

    schema_version: Literal["1"] = "1"
    reviewer_kind: BfclV4SemanticReviewerKind
    principal_id: NonEmptyStr
    human_identity_commitment_sha256: Sha256 | None = None
    model_provider: NonEmptyStr | None = None
    model_name: NonEmptyStr | None = None
    model_revision: NonEmptyStr | None = None
    model_weights_identity_sha256: Sha256 | None = None
    model_prompt_sha256: Sha256 | None = None
    model_generation_seed_u64: Annotated[int, Field(ge=0, le=2**64 - 1, strict=True)] | None = None
    reviewer_kind_declaration_present: Literal[True] = True

    @model_validator(mode="after")
    def _bind_reviewer_kind(self) -> Self:
        model_coordinates = (
            self.model_provider,
            self.model_name,
            self.model_revision,
            self.model_weights_identity_sha256,
            self.model_prompt_sha256,
            self.model_generation_seed_u64,
        )
        if self.reviewer_kind is BfclV4SemanticReviewerKind.HUMAN:
            if self.human_identity_commitment_sha256 is None or any(
                item is not None for item in model_coordinates
            ):
                raise ValueError("human reviewer cannot contain model-assistance coordinates")
        elif self.human_identity_commitment_sha256 is not None or any(
            item is None for item in model_coordinates
        ):
            raise ValueError("model-assisted reviewer requires complete model coordinates")
        return self

    @property
    def independence_key(self) -> str:
        if self.reviewer_kind is BfclV4SemanticReviewerKind.HUMAN:
            payload = {
                "kind": self.reviewer_kind,
                "human_identity_commitment_sha256": self.human_identity_commitment_sha256,
            }
        else:
            payload = {
                "kind": self.reviewer_kind,
                "model_weights_identity_sha256": self.model_weights_identity_sha256,
            }
        return canonical_sha256(payload)


class BfclV4SemanticFamilyAssignment(ImmutableModel):
    """One reviewer-local conservative semantic-family label."""

    audit_item_id: Sha256
    candidate_payload_sha256: Sha256
    semantic_family_label: Annotated[str, Field(min_length=1, max_length=200)]
    rationale: Annotated[str, Field(min_length=1, max_length=2_000)]
    abstained: Literal[False] = False


def _validate_assignment_coverage(
    assignments: tuple[BfclV4SemanticFamilyAssignment, ...],
) -> None:
    expected = tuple(
        (item.audit_item_id, item.candidate_payload_sha256)
        for item in BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items
    )
    observed = tuple((item.audit_item_id, item.candidate_payload_sha256) for item in assignments)
    if observed != expected:
        raise ValueError("semantic-family assignments lack exact ordered universe coverage")


class BfclV4SemanticFamilyReview(ImmutableModel):
    """One declared independent family review; distribution is not authenticated."""

    schema_version: Literal["1"] = "1"
    universe_fingerprint: Sha256
    reviewer: BfclV4SemanticReviewer
    assignments: Annotated[
        tuple[BfclV4SemanticFamilyAssignment, ...], Field(min_length=40, max_length=40)
    ]
    independently_generated_before_pairing: Literal[True] = True
    paired_review_visible_during_generation: Literal[False] = False
    explicit_partition_labels_provided: Literal[False] = False
    partition_roster_lookup_used: Literal[False] = False
    partition_roster_lookup_nonuse_externally_verified: Literal[False] = False
    reviewer_facing_projection_used: Literal[False] = False
    answers_visible: Literal[False] = False
    scores_visible: Literal[False] = False
    holdout_scores_visible: Literal[False] = False
    benchmark_model_outputs_visible: Literal[False] = False
    harness_outputs_visible: Literal[False] = False
    model_campaign_started: Literal[False] = False
    candidate_visible: Literal[False] = False

    @model_validator(mode="after")
    def _bind_review(self) -> Self:
        if self.universe_fingerprint != BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.fingerprint:
            raise ValueError("semantic-family review targets a different universe")
        _validate_assignment_coverage(self.assignments)
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4SemanticAdjudicationDecision(ImmutableModel):
    """Explicit resolution of one pairwise grouping disagreement."""

    left_audit_item_id: Sha256
    right_audit_item_id: Sha256
    same_semantic_family: bool
    rationale: Annotated[str, Field(min_length=1, max_length=2_000)]


class BfclV4SemanticFamilyAdjudication(ImmutableModel):
    """Independent adjudication of every and only double-review disagreement."""

    schema_version: Literal["1"] = "1"
    universe_fingerprint: Sha256
    review_fingerprints: Annotated[tuple[Sha256, Sha256], Field(min_length=2, max_length=2)]
    adjudicator: BfclV4SemanticReviewer
    decisions: tuple[BfclV4SemanticAdjudicationDecision, ...]
    final_assignments: Annotated[
        tuple[BfclV4SemanticFamilyAssignment, ...], Field(min_length=40, max_length=40)
    ]
    reviewer_artifacts_visible: Literal[True] = True
    explicit_partition_labels_provided: Literal[False] = False
    partition_roster_lookup_used: Literal[False] = False
    partition_roster_lookup_nonuse_externally_verified: Literal[False] = False
    reviewer_facing_projection_used: Literal[False] = False
    answers_visible: Literal[False] = False
    scores_visible: Literal[False] = False
    holdout_scores_visible: Literal[False] = False
    benchmark_model_outputs_visible: Literal[False] = False
    harness_outputs_visible: Literal[False] = False
    model_campaign_started: Literal[False] = False
    candidate_visible: Literal[False] = False

    @model_validator(mode="after")
    def _bind_adjudication(self) -> Self:
        if self.universe_fingerprint != BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.fingerprint:
            raise ValueError("semantic-family adjudication targets a different universe")
        _validate_assignment_coverage(self.final_assignments)
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4SemanticAuditEvidence(ImmutableModel):
    """Complete double-review evidence, with adjudication only when required."""

    schema_version: Literal["1"] = "1"
    review_one: BfclV4SemanticFamilyReview
    review_two: BfclV4SemanticFamilyReview
    adjudication: BfclV4SemanticFamilyAdjudication | None = None
    answer_or_score_artifacts_present: Literal[False] = False
    model_campaign_started: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4SemanticAuditVerification(ImmutableModel):
    """Structural verification receipt that cannot attest production provenance."""

    schema_version: Literal["1"] = "1"
    manifest_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    )
    universe_fingerprint: Sha256
    evidence_fingerprint: Sha256
    review_fingerprints: tuple[Sha256, Sha256]
    reviewer_kinds: tuple[BfclV4SemanticReviewerKind, BfclV4SemanticReviewerKind]
    adjudication_fingerprint: Sha256 | None
    disagreement_count: Annotated[int, Field(ge=0, le=780, strict=True)]
    adjudicated_disagreement_count: Annotated[int, Field(ge=0, le=780, strict=True)]
    cross_boundary_family_count: Literal[0] = 0
    exact_population_coverage: Literal[True] = True
    status: Literal["blocked-reviewer-facing-projection-not-implemented"] = (
        "blocked-reviewer-facing-projection-not-implemented"
    )
    reviewer_facing_projection_implemented: Literal[False] = False
    review_distribution_provenance_verified: Literal[False] = False
    partition_inference_prevented: Literal[False] = False
    independent_generation_declared: Literal[True] = True
    independent_generation_externally_verified: Literal[False] = False
    distinct_reviewer_principal_commitments: Literal[True] = True
    reviewer_identity_externally_verified: Literal[False] = False
    review_declarations_authenticated: Literal[False] = False
    same_declared_model_weights_multi_seed_accepted_as_independent: Literal[False] = False
    every_disagreement_adjudicated: Literal[True] = True
    independent_semantic_family_disjointness_attested: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    answers_read: Literal[False] = False
    scores_read: Literal[False] = False
    holdout_scores_read: Literal[False] = False
    model_campaign_started: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_verification(self) -> Self:
        if self.universe_fingerprint != BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.fingerprint:
            raise ValueError("semantic-audit verification targets a different universe")
        if self.adjudicated_disagreement_count != self.disagreement_count:
            raise ValueError("semantic-audit verification leaves a disagreement unresolved")
        if (self.disagreement_count == 0) != (self.adjudication_fingerprint is None):
            raise ValueError("semantic-audit adjudication presence differs from disagreements")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def _make_item(task_id: str) -> BfclV4SemanticAuditItem:
    row_size, row_sha256, candidate_sha256 = _frozen_row_identity(task_id)
    category = _category(task_id)
    return BfclV4SemanticAuditItem(
        audit_item_id=_item_id(task_id, candidate_sha256),
        task_id=task_id,
        question_git_path=(
            f"berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_{category}.json"
        ),
        row_size=row_size,
        row_sha256=row_sha256,
        candidate_payload_sha256=candidate_sha256,
        audit_order_token_sha256=_order_token(task_id),
    )


_ALL_TASK_IDS = BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS + (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS
)
BFCL_V4_SEMANTIC_AUDIT_UNIVERSE = BfclV4SemanticAuditUniverse(
    items=tuple(_make_item(task_id) for task_id in sorted(_ALL_TASK_IDS, key=_order_token))
)
BFCL_V4_SEMANTIC_AUDIT_UNIVERSE_FINGERPRINT = BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.fingerprint


def _relations(
    assignments: tuple[BfclV4SemanticFamilyAssignment, ...],
) -> dict[tuple[str, str], bool]:
    labels = {item.audit_item_id: item.semantic_family_label for item in assignments}
    item_ids = tuple(item.audit_item_id for item in BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items)
    return {
        (left, right): labels[left] == labels[right] for left, right in combinations(item_ids, 2)
    }


def _reject(stage: BfclV4SemanticAuditFailureStage) -> None:
    raise BfclV4SemanticAuditError(stage) from None


def _strict_evidence(
    evidence: BfclV4SemanticAuditEvidence,
) -> BfclV4SemanticAuditEvidence | None:
    try:
        return BfclV4SemanticAuditEvidence.model_validate(evidence, strict=True)
    except Exception:
        return None


def verify_bfcl_v4_semantic_family_audit(
    manifest_fingerprint: str,
    evidence: BfclV4SemanticAuditEvidence,
) -> BfclV4SemanticAuditVerification:
    """Verify independent reviews, complete arbitration, and boundary isolation."""

    checked = _strict_evidence(evidence)
    if (
        checked is None
        or manifest_fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    ):
        _reject(BfclV4SemanticAuditFailureStage.EVIDENCE_BINDING)

    first, second = checked.review_one, checked.review_two
    if (
        first.fingerprint == second.fingerprint
        or first.reviewer.principal_id == second.reviewer.principal_id
        or first.reviewer.independence_key == second.reviewer.independence_key
    ):
        _reject(BfclV4SemanticAuditFailureStage.REVIEW_INDEPENDENCE)

    first_relations, second_relations = (
        _relations(first.assignments),
        _relations(second.assignments),
    )
    disagreements = tuple(
        pair for pair in first_relations if first_relations[pair] != second_relations[pair]
    )
    adjudication = checked.adjudication
    if bool(disagreements) != (adjudication is not None):
        _reject(BfclV4SemanticAuditFailureStage.DISAGREEMENT_ARBITRATION)

    if adjudication is None:
        final_assignments = first.assignments
        adjudication_fingerprint = None
    else:
        if (
            adjudication.review_fingerprints != (first.fingerprint, second.fingerprint)
            or adjudication.adjudicator.principal_id
            in {first.reviewer.principal_id, second.reviewer.principal_id}
            or adjudication.adjudicator.independence_key
            in {first.reviewer.independence_key, second.reviewer.independence_key}
        ):
            _reject(BfclV4SemanticAuditFailureStage.REVIEW_INDEPENDENCE)
        decision_pairs = tuple(
            (item.left_audit_item_id, item.right_audit_item_id) for item in adjudication.decisions
        )
        if decision_pairs != disagreements:
            _reject(BfclV4SemanticAuditFailureStage.DISAGREEMENT_ARBITRATION)
        final_relations = _relations(adjudication.final_assignments)
        decisions = {
            (item.left_audit_item_id, item.right_audit_item_id): item.same_semantic_family
            for item in adjudication.decisions
        }
        for pair, final_same in final_relations.items():
            expected = decisions[pair] if pair in decisions else first_relations[pair]
            if final_same != expected:
                _reject(BfclV4SemanticAuditFailureStage.DISAGREEMENT_ARBITRATION)
        final_assignments = adjudication.final_assignments
        adjudication_fingerprint = adjudication.fingerprint

    boundary_by_task_id = {
        **{task_id: "v1" for task_id in BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS},
        **{
            item.task_id: item.split.value for item in BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.roster
        },
    }
    task_by_audit_id = {
        item.audit_item_id: item.task_id for item in BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items
    }
    for (left, right), same_family in _relations(final_assignments).items():
        if same_family and (
            boundary_by_task_id[task_by_audit_id[left]]
            != boundary_by_task_id[task_by_audit_id[right]]
        ):
            _reject(BfclV4SemanticAuditFailureStage.FAMILY_BOUNDARY)

    return BfclV4SemanticAuditVerification(
        universe_fingerprint=BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.fingerprint,
        evidence_fingerprint=checked.fingerprint,
        review_fingerprints=(first.fingerprint, second.fingerprint),
        reviewer_kinds=(first.reviewer.reviewer_kind, second.reviewer.reviewer_kind),
        adjudication_fingerprint=adjudication_fingerprint,
        disagreement_count=len(disagreements),
        adjudicated_disagreement_count=len(disagreements),
    )


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_") or name.startswith("Bfcl") or name.startswith("verify_bfcl")
]
