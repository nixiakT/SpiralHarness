from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.bfcl_v4_public_development_v2 as development_v2_subject
from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    BfclV4PublicDevelopmentV2ExecutionError,
    assert_bfcl_v4_public_development_v2_score_bearing_execution_allowed,
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
    BfclV4PublicDevelopmentV2Split,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_semantic_audit import (
    BFCL_V4_SEMANTIC_AUDIT_UNIVERSE,
    BFCL_V4_SEMANTIC_AUDIT_UNIVERSE_FINGERPRINT,
    BfclV4SemanticAdjudicationDecision,
    BfclV4SemanticAuditError,
    BfclV4SemanticAuditEvidence,
    BfclV4SemanticAuditFailureStage,
    BfclV4SemanticFamilyAdjudication,
    BfclV4SemanticFamilyAssignment,
    BfclV4SemanticFamilyReview,
    BfclV4SemanticReviewer,
    BfclV4SemanticReviewerKind,
    verify_bfcl_v4_semantic_family_audit,
)

_DEFAULT_PINNED_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")


@pytest.fixture(scope="module")
def loaded():
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_PINNED_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for semantic guard test")
    return load_bfcl_v4_public_development_v2(checkout)


def _human(principal: str, marker: str) -> BfclV4SemanticReviewer:
    return BfclV4SemanticReviewer(
        reviewer_kind=BfclV4SemanticReviewerKind.HUMAN,
        principal_id=principal,
        human_identity_commitment_sha256=marker * 64,
    )


def _model(principal: str, seed: int) -> BfclV4SemanticReviewer:
    return BfclV4SemanticReviewer(
        reviewer_kind=BfclV4SemanticReviewerKind.MODEL_ASSISTED,
        principal_id=principal,
        model_provider="open-provider",
        model_name="same-open-model",
        model_revision="frozen-revision",
        model_weights_identity_sha256="e" * 64,
        model_prompt_sha256="d" * 64,
        model_generation_seed_u64=seed,
    )


def _assignments(
    prefix: str,
    *,
    merged_pairs: tuple[tuple[str, str], ...] = (),
) -> tuple[BfclV4SemanticFamilyAssignment, ...]:
    labels = {
        item.audit_item_id: f"{prefix}-family-{index}"
        for index, item in enumerate(BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items)
    }
    for left, right in merged_pairs:
        labels[right] = labels[left]
    return tuple(
        BfclV4SemanticFamilyAssignment(
            audit_item_id=item.audit_item_id,
            candidate_payload_sha256=item.candidate_payload_sha256,
            semantic_family_label=labels[item.audit_item_id],
            rationale=f"Independent conservative question-only rationale for {item.audit_item_id}.",
        )
        for item in BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items
    )


def _review(
    reviewer: BfclV4SemanticReviewer,
    prefix: str,
    *,
    merged_pairs: tuple[tuple[str, str], ...] = (),
) -> BfclV4SemanticFamilyReview:
    return BfclV4SemanticFamilyReview(
        universe_fingerprint=BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.fingerprint,
        reviewer=reviewer,
        assignments=_assignments(prefix, merged_pairs=merged_pairs),
    )


def _consensus_evidence() -> BfclV4SemanticAuditEvidence:
    return BfclV4SemanticAuditEvidence(
        review_one=_review(_human("reviewer-one", "1"), "one"),
        review_two=_review(_human("reviewer-two", "2"), "two"),
    )


def _single_disagreement_reviews() -> tuple[
    BfclV4SemanticFamilyReview,
    BfclV4SemanticFamilyReview,
    str,
    str,
]:
    first = _review(_human("reviewer-one", "1"), "one")
    left, right = (
        BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items[0].audit_item_id,
        BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items[1].audit_item_id,
    )
    second = _review(
        _human("reviewer-two", "2"),
        "two",
        merged_pairs=((left, right),),
    )
    return first, second, left, right


def _decision(
    left: str,
    right: str,
    *,
    same_semantic_family: bool = False,
) -> BfclV4SemanticAdjudicationDecision:
    return BfclV4SemanticAdjudicationDecision(
        left_audit_item_id=left,
        right_audit_item_id=right,
        same_semantic_family=same_semantic_family,
        rationale="The two question intents and tool semantics were compared explicitly.",
    )


def _adjudication(
    first: BfclV4SemanticFamilyReview,
    second: BfclV4SemanticFamilyReview,
    *,
    adjudicator: BfclV4SemanticReviewer,
    decisions: tuple[BfclV4SemanticAdjudicationDecision, ...],
    final_assignments: tuple[BfclV4SemanticFamilyAssignment, ...],
) -> BfclV4SemanticFamilyAdjudication:
    return BfclV4SemanticFamilyAdjudication(
        universe_fingerprint=BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.fingerprint,
        review_fingerprints=(first.fingerprint, second.fingerprint),
        adjudicator=adjudicator,
        decisions=decisions,
        final_assignments=final_assignments,
    )


def test_universe_is_exact_question_only_trusted_binding_population() -> None:
    universe = BFCL_V4_SEMANTIC_AUDIT_UNIVERSE

    assert universe.fingerprint == BFCL_V4_SEMANTIC_AUDIT_UNIVERSE_FINGERPRINT
    assert universe.manifest_fingerprint == BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    assert len(universe.items) == 40
    assert len({item.task_id for item in universe.items}) == 40
    assert len({item.audit_item_id for item in universe.items}) == 40
    assert universe.explicit_partition_label_field_present is False
    assert universe.task_ids_and_paths_present is True
    assert universe.partition_inference_prevented is False
    assert universe.trusted_control_plane_only is True
    assert universe.reviewer_facing is False
    assert universe.reviewer_facing_projection_implemented is False
    assert universe.answers_present is False
    assert universe.scores_present is False
    assert universe.holdout_scores_present is False
    assert universe.model_or_harness_results_present is False
    assert all("split" not in item.model_dump(mode="json") for item in universe.items)
    assert all(item.task_id and item.question_git_path for item in universe.items)


def test_human_and_model_assisted_reviewer_types_fail_closed() -> None:
    assert _human("declared-human", "a").reviewer_kind_declaration_present is True
    assert _model("declared-model", 1).reviewer_kind_declaration_present is True
    with pytest.raises(ValidationError, match="human reviewer cannot contain model-assistance"):
        BfclV4SemanticReviewer(
            reviewer_kind=BfclV4SemanticReviewerKind.HUMAN,
            principal_id="false-human",
            human_identity_commitment_sha256="a" * 64,
            model_provider="provider",
            model_name="model",
            model_revision="revision",
            model_weights_identity_sha256="c" * 64,
            model_prompt_sha256="b" * 64,
            model_generation_seed_u64=7,
        )
    with pytest.raises(ValidationError, match="complete model coordinates"):
        BfclV4SemanticReviewer(
            reviewer_kind=BfclV4SemanticReviewerKind.MODEL_ASSISTED,
            principal_id="incomplete-model",
        )


def test_same_model_multiple_seeds_are_not_independent_reviewers() -> None:
    evidence = BfclV4SemanticAuditEvidence(
        review_one=_review(_model("model-seed-one", 1), "one"),
        review_two=_review(_model("model-seed-two", 2), "two"),
    )

    with pytest.raises(BfclV4SemanticAuditError) as captured:
        verify_bfcl_v4_semantic_family_audit(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            evidence,
        )
    assert captured.value.failure_stage is BfclV4SemanticAuditFailureStage.REVIEW_INDEPENDENCE
    assert captured.value.__context__ is None
    assert captured.value.__cause__ is None


def test_consensus_structure_passes_but_production_distribution_remains_blocked() -> None:
    evidence = _consensus_evidence()
    verification = verify_bfcl_v4_semantic_family_audit(
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
        evidence,
    )

    assert verification.evidence_fingerprint == evidence.fingerprint
    assert verification.reviewer_kinds == (
        BfclV4SemanticReviewerKind.HUMAN,
        BfclV4SemanticReviewerKind.HUMAN,
    )
    assert verification.disagreement_count == 0
    assert verification.adjudication_fingerprint is None
    assert verification.status == "blocked-reviewer-facing-projection-not-implemented"
    assert verification.reviewer_facing_projection_implemented is False
    assert verification.review_distribution_provenance_verified is False
    assert verification.partition_inference_prevented is False
    assert verification.independent_generation_declared is True
    assert verification.independent_generation_externally_verified is False
    assert verification.distinct_reviewer_principal_commitments is True
    assert verification.reviewer_identity_externally_verified is False
    assert verification.review_declarations_authenticated is False
    assert verification.independent_semantic_family_disjointness_attested is False
    assert verification.same_declared_model_weights_multi_seed_accepted_as_independent is False
    assert verification.score_bearing_execution_allowed is False
    assert verification.answers_read is False
    assert verification.holdout_scores_read is False
    assert verification.model_campaign_started is False
    assert evidence.review_one.partition_roster_lookup_used is False
    assert evidence.review_one.partition_roster_lookup_nonuse_externally_verified is False
    assert evidence.review_one.reviewer_facing_projection_used is False
    assert len(verification.fingerprint) == 64


def test_every_disagreement_requires_exact_independent_adjudication() -> None:
    first, second, left, right = _single_disagreement_reviews()
    unadjudicated = BfclV4SemanticAuditEvidence(review_one=first, review_two=second)
    with pytest.raises(BfclV4SemanticAuditError) as captured:
        verify_bfcl_v4_semantic_family_audit(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            unadjudicated,
        )
    assert captured.value.failure_stage is BfclV4SemanticAuditFailureStage.DISAGREEMENT_ARBITRATION

    adjudication = _adjudication(
        first,
        second,
        adjudicator=_human("adjudicator", "3"),
        decisions=(_decision(left, right),),
        final_assignments=_assignments("final"),
    )
    verification = verify_bfcl_v4_semantic_family_audit(
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
        BfclV4SemanticAuditEvidence(
            review_one=first,
            review_two=second,
            adjudication=adjudication,
        ),
    )
    assert verification.disagreement_count == 1
    assert verification.adjudicated_disagreement_count == 1
    assert verification.adjudication_fingerprint == adjudication.fingerprint
    assert verification.independent_semantic_family_disjointness_attested is False
    assert verification.score_bearing_execution_allowed is False
    assert adjudication.partition_roster_lookup_used is False
    assert adjudication.partition_roster_lookup_nonuse_externally_verified is False


def test_adjudication_decisions_must_equal_the_exact_disagreement_set() -> None:
    first, second, left, right = _single_disagreement_reviews()
    extra_left, extra_right = (
        BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items[2].audit_item_id,
        BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items[3].audit_item_id,
    )
    decision_cases = {
        "missing": (),
        "wrong": (_decision(right, left),),
        "extra": (_decision(left, right), _decision(extra_left, extra_right)),
    }

    for case, decisions in decision_cases.items():
        evidence = BfclV4SemanticAuditEvidence(
            review_one=first,
            review_two=second,
            adjudication=_adjudication(
                first,
                second,
                adjudicator=_human("adjudicator", "3"),
                decisions=decisions,
                final_assignments=_assignments("final"),
            ),
        )
        with pytest.raises(BfclV4SemanticAuditError) as captured:
            verify_bfcl_v4_semantic_family_audit(
                BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
                evidence,
            )
        assert (
            captured.value.failure_stage is BfclV4SemanticAuditFailureStage.DISAGREEMENT_ARBITRATION
        ), case


def test_same_principal_adjudicator_is_not_independent() -> None:
    first, second, left, right = _single_disagreement_reviews()
    evidence = BfclV4SemanticAuditEvidence(
        review_one=first,
        review_two=second,
        adjudication=_adjudication(
            first,
            second,
            adjudicator=_human("reviewer-one", "3"),
            decisions=(_decision(left, right),),
            final_assignments=_assignments("final"),
        ),
    )

    with pytest.raises(BfclV4SemanticAuditError) as captured:
        verify_bfcl_v4_semantic_family_audit(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            evidence,
        )
    assert captured.value.failure_stage is BfclV4SemanticAuditFailureStage.REVIEW_INDEPENDENCE


def test_adjudication_final_assignments_must_implement_each_decision() -> None:
    first, second, left, right = _single_disagreement_reviews()
    evidence = BfclV4SemanticAuditEvidence(
        review_one=first,
        review_two=second,
        adjudication=_adjudication(
            first,
            second,
            adjudicator=_human("adjudicator", "3"),
            decisions=(_decision(left, right, same_semantic_family=False),),
            final_assignments=_assignments("final", merged_pairs=((left, right),)),
        ),
    )

    with pytest.raises(BfclV4SemanticAuditError) as captured:
        verify_bfcl_v4_semantic_family_audit(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            evidence,
        )
    assert captured.value.failure_stage is BfclV4SemanticAuditFailureStage.DISAGREEMENT_ARBITRATION


def test_cross_boundary_semantic_family_rejects_even_with_double_consensus() -> None:
    fit_task = next(
        item.task_id
        for item in BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.roster
        if item.split is BfclV4PublicDevelopmentV2Split.FIT
    )
    holdout_task = next(
        item.task_id
        for item in BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.roster
        if item.split is BfclV4PublicDevelopmentV2Split.HOLDOUT
    )
    by_task = {item.task_id: item.audit_item_id for item in BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items}
    merged = ((by_task[fit_task], by_task[holdout_task]),)
    evidence = BfclV4SemanticAuditEvidence(
        review_one=_review(_human("reviewer-one", "1"), "one", merged_pairs=merged),
        review_two=_review(_human("reviewer-two", "2"), "two", merged_pairs=merged),
    )

    with pytest.raises(BfclV4SemanticAuditError) as captured:
        verify_bfcl_v4_semantic_family_audit(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            evidence,
        )
    assert captured.value.failure_stage is BfclV4SemanticAuditFailureStage.FAMILY_BOUNDARY


def test_forged_review_fails_with_context_free_typed_rejection() -> None:
    evidence = _consensus_evidence()
    assignments = list(evidence.review_one.assignments)
    assignments[0] = assignments[0].model_copy(update={"candidate_payload_sha256": "f" * 64})
    forged_review = BfclV4SemanticFamilyReview.model_construct(
        **{
            **evidence.review_one.model_dump(mode="python"),
            "assignments": tuple(assignments),
        }
    )
    forged = BfclV4SemanticAuditEvidence.model_construct(
        review_one=forged_review,
        review_two=evidence.review_two,
        adjudication=None,
    )

    with pytest.raises(BfclV4SemanticAuditError) as captured:
        verify_bfcl_v4_semantic_family_audit(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            forged,
        )
    assert captured.value.failure_stage is BfclV4SemanticAuditFailureStage.EVIDENCE_BINDING
    assert captured.value.__context__ is None
    assert captured.value.__cause__ is None


def test_provider_free_semantic_evidence_guard_remains_closed() -> None:
    with pytest.raises(
        BfclV4PublicDevelopmentV2ExecutionError,
        match=(
            "reviewer-facing question/schema projection, review-distribution provenance, "
            "and external reviewer-authority authentication are not verified"
        ),
    ):
        development_v2_subject._assert_bfcl_v4_public_development_v2_semantic_evidence_allows_execution(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            _consensus_evidence(),
        )


def test_score_guard_requires_evidence_and_still_cannot_release_campaign(loaded) -> None:
    with pytest.raises(
        BfclV4PublicDevelopmentV2ExecutionError,
        match=(
            "reviewer-facing question/schema projection, review-distribution provenance, "
            "and external reviewer-authority authentication are not verified"
        ),
    ):
        assert_bfcl_v4_public_development_v2_score_bearing_execution_allowed(
            loaded,
            _consensus_evidence(),
        )
