"""Question-only loader and fail-closed audit for the prospective BFCL v2 roster."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import spiral_harness.benchmark.bfcl_v4_question_source as question_source
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CANDIDATE_POOL_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_TOTALS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_ELIGIBLE_SINGLETONS_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_LEXICAL_AUDIT_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SOURCE_BINDINGS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SPLIT_CATEGORY_QUOTAS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_SHA256,
    BFCL_V4_UPSTREAM_COMMIT,
    BfclV4LoadedPublicDevelopmentV2,
    BfclV4PublicDevelopmentV2AuditReceipt,
    BfclV4PublicDevelopmentV2AuditStatus,
    BfclV4PublicDevelopmentV2FailureStage,
    BfclV4PublicDevelopmentV2RosterEntry,
    BfclV4PublicDevelopmentV2Split,
    BfclV4PublicDevelopmentV2Task,
    bfcl_v4_public_development_v2_selection_token,
    bfcl_v4_public_development_v2_v1_exclusion_sha256,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_identities import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_QUESTION_BLOB_IDENTITIES,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_identities import (
    BFCL_V4_PILOT_ROW_IDENTITIES,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256

_CATEGORY_ROW_COUNTS = {
    "simple_python": 400,
    "multiple": 200,
    "parallel": 200,
    "parallel_multiple": 200,
}
_CANDIDATE_POOL_DOMAIN = "spiral-bfcl-v4-public-development-v2-candidate-pool/v1"
_ELIGIBLE_DOMAIN = "spiral-bfcl-v4-public-development-v2-eligible-singletons/v1"
_LEXICAL_AUDIT_DOMAIN = "spiral-bfcl-v4-public-development-v2-lexical-near-duplicate-audit/v1"


class BfclV4PublicDevelopmentV2Error(RuntimeError):
    """A v2 roster audit rejection carrying a typed, credential-free receipt."""

    def __init__(self, receipt: BfclV4PublicDevelopmentV2AuditReceipt) -> None:
        self.receipt = receipt
        super().__init__(f"BFCL v2 prospective roster rejected at {receipt.failure_stage.value}")


class BfclV4PublicDevelopmentV2ExecutionError(RuntimeError):
    """The public-development roster cannot cross a score-bearing boundary."""


_Candidate = question_source.BfclV4QuestionCandidate


@dataclass(frozen=True, slots=True)
class _Selected:
    candidate: _Candidate
    split: BfclV4PublicDevelopmentV2Split
    selection_token_sha256: str


def _capture[CapturedT](
    operation: Callable[[], CapturedT],
) -> tuple[bool, CapturedT | None]:
    """Erase a lower-level exception before the caller emits a typed rejection."""

    try:
        return True, operation()
    except Exception:
        return False, None


def _candidate_pool_sha256(candidates: tuple[_Candidate, ...]) -> str:
    return canonical_sha256(
        {
            "domain": _CANDIDATE_POOL_DOMAIN,
            "rows": tuple(
                (
                    item.category,
                    item.task_id,
                    item.row_sha256,
                    item.candidate_payload_sha256,
                    item.source_family_sha256,
                    item.semantic_template_sha256,
                )
                for item in candidates
            ),
        }
    )


def _family_components(candidates: tuple[_Candidate, ...]) -> tuple[int, ...]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_name: dict[str, int] = {}
    by_template: dict[tuple[str, ...], int] = {}
    for index, candidate in enumerate(candidates):
        for name in candidate.official_function_names:
            previous = by_name.setdefault(name, index)
            union(index, previous)
        previous_template = by_template.setdefault(candidate.semantic_tokens, index)
        union(index, previous_template)
    return tuple(find(index) for index in range(len(candidates)))


def _eligible_singletons(
    candidates: tuple[_Candidate, ...],
    components: tuple[int, ...],
) -> tuple[_Candidate, ...]:
    sizes: dict[int, int] = defaultdict(int)
    for component in components:
        sizes[component] += 1
    v1_ids = set(BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS)
    return tuple(
        candidate
        for candidate, component in zip(candidates, components, strict=True)
        if sizes[component] == 1 and candidate.task_id not in v1_ids
    )


def _eligible_sha256(eligible: tuple[_Candidate, ...]) -> str:
    return canonical_sha256(
        {
            "domain": _ELIGIBLE_DOMAIN,
            "task_ids": tuple(sorted(item.task_id for item in eligible)),
        }
    )


def _select(eligible: tuple[_Candidate, ...]) -> tuple[_Selected, ...]:
    per_category: dict[str, tuple[tuple[str, _Candidate], ...]] = {}
    for category in BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER:
        ranked = sorted(
            (
                (
                    bfcl_v4_public_development_v2_selection_token(
                        category,
                        candidate.task_id,
                    ),
                    candidate,
                )
                for candidate in eligible
                if candidate.category == category
            ),
            key=lambda item: (item[0], item[1].task_id),
        )
        total = BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_TOTALS[category]
        if len(ranked) < total:
            raise ValueError("eligible singleton pool cannot fill a category quota")
        per_category[category] = tuple(ranked[:total])

    selected: list[_Selected] = []
    offsets = {category: 0 for category in BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER}
    for split in BfclV4PublicDevelopmentV2Split:
        for category in BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER:
            count = BFCL_V4_PUBLIC_DEVELOPMENT_V2_SPLIT_CATEGORY_QUOTAS[split.value][category]
            start = offsets[category]
            for token, candidate in per_category[category][start : start + count]:
                selected.append(
                    _Selected(
                        candidate=candidate,
                        split=split,
                        selection_token_sha256=token,
                    )
                )
            offsets[category] += count
    return tuple(selected)


def _shingles(tokens: tuple[str, ...]) -> frozenset[tuple[str, ...]]:
    if len(tokens) < 3:
        return frozenset((tokens,))
    return frozenset(tokens[index : index + 3] for index in range(len(tokens) - 2))


def _lexical_comparison(
    left: _Selected,
    right: _Candidate,
    *,
    right_boundary: str,
) -> dict[str, object]:
    left_shingles = _shingles(left.candidate.semantic_tokens)
    right_shingles = _shingles(right.semantic_tokens)
    intersection = len(left_shingles & right_shingles)
    union = len(left_shingles | right_shingles)
    minimum = min(len(left_shingles), len(right_shingles))
    flagged = 100 * intersection >= 67 * union and 5 * intersection >= 4 * minimum
    return {
        "left_task_id": left.candidate.task_id,
        "left_boundary": left.split.value,
        "right_task_id": right.task_id,
        "right_boundary": right_boundary,
        "intersection": intersection,
        "union": union,
        "minimum": minimum,
        "flagged": flagged,
    }


def _lexical_audit(
    selected: tuple[_Selected, ...],
    by_id: dict[str, _Candidate],
) -> tuple[str, int, int]:
    comparisons: list[dict[str, object]] = []
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            if left.split is right.split:
                continue
            comparisons.append(
                _lexical_comparison(
                    left,
                    right.candidate,
                    right_boundary=right.split.value,
                )
            )
    for left in selected:
        for task_id in BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS:
            comparisons.append(_lexical_comparison(left, by_id[task_id], right_boundary="v1"))
    digest = canonical_sha256(
        {
            "domain": _LEXICAL_AUDIT_DOMAIN,
            "jaccard_threshold": {"numerator": 67, "denominator": 100},
            "containment_threshold": {"numerator": 4, "denominator": 5},
            "shingle_width": 3,
            "comparisons": tuple(comparisons),
        }
    )
    return digest, len(comparisons), sum(bool(item["flagged"]) for item in comparisons)


def _function_name_audit(
    candidates: tuple[_Candidate, ...],
    components: tuple[int, ...],
    selected: tuple[_Selected, ...],
) -> tuple[dict[str, _Candidate], int, int, int]:
    by_id = {item.task_id: item for item in candidates}
    selected_ids = {item.candidate.task_id for item in selected}
    v1_ids = set(BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS)
    roots_by_id = {
        candidate.task_id: root for candidate, root in zip(candidates, components, strict=True)
    }
    v1_roots = {roots_by_id[task_id] for task_id in v1_ids}
    family_overlap = sum(roots_by_id[task_id] in v1_roots for task_id in selected_ids)
    official_overlap = 0
    for index, left in enumerate(selected):
        left_names = set(left.candidate.official_function_names)
        for right in selected[index + 1 :]:
            if left.split is not right.split:
                official_overlap += len(left_names & set(right.candidate.official_function_names))
        for task_id in BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS:
            official_overlap += len(left_names & set(by_id[task_id].official_function_names))
    return by_id, len(v1_ids & selected_ids), family_overlap, official_overlap


def _rejected(
    stage: BfclV4PublicDevelopmentV2FailureStage,
    *,
    source_bindings: tuple[question_source.BfclV4QuestionSourceBinding, ...] = (),
    candidate_pool_sha256: str | None = None,
    eligible_singletons_sha256: str | None = None,
    lexical_audit_sha256: str | None = None,
    attempted_selected_task_ids: tuple[str, ...] = (),
    source_row_count: int = 0,
    eligible_singleton_count: int = 0,
    lexical_comparison_count: int = 0,
    lexical_near_duplicate_count: int = 0,
    official_function_name_overlap_count: int = 0,
    v1_task_overlap_count: int = 0,
    v1_family_overlap_count: int = 0,
) -> NoReturn:
    receipt = BfclV4PublicDevelopmentV2AuditReceipt(
        status=BfclV4PublicDevelopmentV2AuditStatus.REJECTED,
        failure_stage=stage,
        source_bindings=source_bindings,
        candidate_pool_sha256=candidate_pool_sha256,
        eligible_singletons_sha256=eligible_singletons_sha256,
        lexical_audit_sha256=lexical_audit_sha256,
        attempted_selected_task_ids=attempted_selected_task_ids,
        source_row_count=source_row_count,
        eligible_singleton_count=eligible_singleton_count,
        selected_task_count=len(attempted_selected_task_ids),
        lexical_comparison_count=lexical_comparison_count,
        lexical_near_duplicate_count=lexical_near_duplicate_count,
        official_function_name_overlap_count=official_function_name_overlap_count,
        v1_task_overlap_count=v1_task_overlap_count,
        v1_family_overlap_count=v1_family_overlap_count,
    )
    raise BfclV4PublicDevelopmentV2Error(receipt) from None


def load_bfcl_v4_public_development_v2(
    checkout: str | Path,
) -> BfclV4LoadedPublicDevelopmentV2:
    """Recompute, audit, and load the frozen question-only v2 roster."""

    if question_source.BFCL_V4_QUESTION_SOURCE_UPSTREAM_COMMIT != BFCL_V4_UPSTREAM_COMMIT:
        _rejected(BfclV4PublicDevelopmentV2FailureStage.CHECKOUT_LINEAGE)
    if tuple(BFCL_V4_PILOT_ROW_IDENTITIES) != BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS:
        _rejected(BfclV4PublicDevelopmentV2FailureStage.V1_LINEAGE)

    success, source_checkout = _capture(
        lambda: question_source.open_bfcl_v4_question_checkout(checkout)
    )
    if not success or source_checkout is None:
        _rejected(BfclV4PublicDevelopmentV2FailureStage.CHECKOUT_LINEAGE)

    success, v1_exclusion_sha256 = _capture(
        lambda: bfcl_v4_public_development_v2_v1_exclusion_sha256(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS
        )
    )
    if not success or v1_exclusion_sha256 != BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_SHA256:
        _rejected(BfclV4PublicDevelopmentV2FailureStage.V1_LINEAGE)

    source_bindings: list[question_source.BfclV4QuestionSourceBinding] = []
    candidates: list[_Candidate] = []
    for category, expected_binding in zip(
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CATEGORY_ORDER,
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_SOURCE_BINDINGS,
        strict=True,
    ):
        success, source = _capture(
            lambda expected_binding=expected_binding: question_source.read_bfcl_v4_question_blob(
                source_checkout,
                expected_binding.git_path,
            )
        )
        if not success or source is None:
            _rejected(
                BfclV4PublicDevelopmentV2FailureStage.QUESTION_SOURCE,
                source_bindings=tuple(source_bindings),
            )
        content, observed_binding = source
        if observed_binding != expected_binding:
            _rejected(
                BfclV4PublicDevelopmentV2FailureStage.QUESTION_SOURCE,
                source_bindings=(*source_bindings, observed_binding),
            )
        source_bindings.append(observed_binding)
        success, parsed = _capture(
            lambda category=category, content=content: question_source.parse_bfcl_v4_question_blob(
                category,
                content,
                expected_row_count=_CATEGORY_ROW_COUNTS[category],
            )
        )
        if not success or parsed is None:
            _rejected(
                BfclV4PublicDevelopmentV2FailureStage.SOURCE_PARSE,
                source_bindings=tuple(source_bindings),
                source_row_count=len(candidates),
            )
        candidates.extend(parsed)

    frozen_bindings = tuple(source_bindings)
    frozen_candidates = tuple(candidates)
    success, candidate_pool_sha256 = _capture(lambda: _candidate_pool_sha256(frozen_candidates))
    if not success or candidate_pool_sha256 != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CANDIDATE_POOL_SHA256:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FROZEN_IDENTITY,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            source_row_count=len(frozen_candidates),
        )

    success, components = _capture(lambda: _family_components(frozen_candidates))
    if not success or components is None:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FAMILY_GRAPH,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            source_row_count=len(frozen_candidates),
        )
    success, eligible = _capture(lambda: _eligible_singletons(frozen_candidates, components))
    if not success or eligible is None:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FAMILY_GRAPH,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            source_row_count=len(frozen_candidates),
        )
    success, eligible_sha256 = _capture(lambda: _eligible_sha256(eligible))
    if not success or eligible_sha256 != BFCL_V4_PUBLIC_DEVELOPMENT_V2_ELIGIBLE_SINGLETONS_SHA256:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FAMILY_GRAPH,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
        )

    success, selected = _capture(lambda: _select(eligible))
    if not success or selected is None:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.DETERMINISTIC_SELECTION,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
        )
    selected_ids = tuple(item.candidate.task_id for item in selected)
    if selected_ids != BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.DETERMINISTIC_SELECTION,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            attempted_selected_task_ids=selected_ids,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
        )

    success, function_audit = _capture(
        lambda: _function_name_audit(frozen_candidates, components, selected)
    )
    if not success or function_audit is None:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FUNCTION_NAME_AUDIT,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            attempted_selected_task_ids=selected_ids,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
        )
    by_id, v1_task_overlap_count, v1_family_overlap_count, official_overlap_count = function_audit
    if v1_task_overlap_count or v1_family_overlap_count or official_overlap_count:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FUNCTION_NAME_AUDIT,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            attempted_selected_task_ids=selected_ids,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
            official_function_name_overlap_count=official_overlap_count,
            v1_task_overlap_count=v1_task_overlap_count,
            v1_family_overlap_count=v1_family_overlap_count,
        )

    success, lexical_audit = _capture(lambda: _lexical_audit(selected, by_id))
    if not success or lexical_audit is None:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.NEAR_DUPLICATE_AUDIT,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            attempted_selected_task_ids=selected_ids,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
        )
    lexical_sha256, comparison_count, near_duplicate_count = lexical_audit
    if (
        lexical_sha256 != BFCL_V4_PUBLIC_DEVELOPMENT_V2_LEXICAL_AUDIT_SHA256
        or comparison_count != 539
        or near_duplicate_count
    ):
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.NEAR_DUPLICATE_AUDIT,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            lexical_audit_sha256=lexical_sha256,
            attempted_selected_task_ids=selected_ids,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
            lexical_comparison_count=comparison_count,
            lexical_near_duplicate_count=near_duplicate_count,
        )

    success, entries = _capture(
        lambda: tuple(
            BfclV4PublicDevelopmentV2RosterEntry(
                ordinal=ordinal,
                task_id=item.candidate.task_id,
                category=item.candidate.category,
                split=item.split,
                official_function_names=item.candidate.official_function_names,
                source_family_sha256=item.candidate.source_family_sha256,
                semantic_template_sha256=item.candidate.semantic_template_sha256,
                selection_token_sha256=item.selection_token_sha256,
                question_git_path=(
                    "berkeley-function-call-leaderboard/bfcl_eval/data/"
                    f"BFCL_v4_{item.candidate.category}.json"
                ),
                question_blob_size=(
                    BFCL_V4_PUBLIC_DEVELOPMENT_V2_QUESTION_BLOB_IDENTITIES[item.candidate.category][
                        0
                    ]
                ),
                question_blob_sha256=(
                    BFCL_V4_PUBLIC_DEVELOPMENT_V2_QUESTION_BLOB_IDENTITIES[item.candidate.category][
                        1
                    ]
                ),
                row_size=len(item.candidate.row),
                row_sha256=item.candidate.row_sha256,
                candidate_payload_sha256=item.candidate.candidate_payload_sha256,
            )
            for ordinal, item in enumerate(selected)
        )
    )
    if (
        not success
        or entries != BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.roster
        or (
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.fingerprint
            != BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
        )
    ):
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FROZEN_IDENTITY,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            lexical_audit_sha256=lexical_sha256,
            attempted_selected_task_ids=selected_ids,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
            lexical_comparison_count=comparison_count,
        )

    success, tasks = _capture(
        lambda: tuple(
            BfclV4PublicDevelopmentV2Task(
                task_id=item.candidate.task_id,
                category=item.candidate.category,
                question_json=canonical_json(item.candidate.value["question"]),
                function_schemas_json=canonical_json(item.candidate.value["function"]),
                official_function_names=item.candidate.official_function_names,
                candidate_payload_sha256=item.candidate.candidate_payload_sha256,
            )
            for item in selected
        )
    )
    if not success or tasks is None:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FROZEN_IDENTITY,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            lexical_audit_sha256=lexical_sha256,
            attempted_selected_task_ids=selected_ids,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
            lexical_comparison_count=comparison_count,
        )
    success, receipt = _capture(
        lambda: BfclV4PublicDevelopmentV2AuditReceipt(
            status=BfclV4PublicDevelopmentV2AuditStatus.ACCEPTED,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            lexical_audit_sha256=lexical_sha256,
            attempted_selected_task_ids=selected_ids,
            manifest_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.fingerprint,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
            selected_task_count=len(selected),
            lexical_comparison_count=comparison_count,
            lexical_near_duplicate_count=near_duplicate_count,
            official_function_name_overlap_count=official_overlap_count,
            v1_task_overlap_count=v1_task_overlap_count,
            v1_family_overlap_count=v1_family_overlap_count,
        )
    )
    if not success or receipt is None:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FROZEN_IDENTITY,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            lexical_audit_sha256=lexical_sha256,
            attempted_selected_task_ids=selected_ids,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
            lexical_comparison_count=comparison_count,
        )
    success, loaded = _capture(
        lambda: BfclV4LoadedPublicDevelopmentV2(
            manifest=BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST,
            tasks=tasks,
            audit_receipt=receipt,
        )
    )
    if not success or loaded is None:
        _rejected(
            BfclV4PublicDevelopmentV2FailureStage.FROZEN_IDENTITY,
            source_bindings=frozen_bindings,
            candidate_pool_sha256=candidate_pool_sha256,
            eligible_singletons_sha256=eligible_sha256,
            lexical_audit_sha256=lexical_sha256,
            attempted_selected_task_ids=selected_ids,
            source_row_count=len(frozen_candidates),
            eligible_singleton_count=len(eligible),
            lexical_comparison_count=comparison_count,
        )
    return loaded


def assert_bfcl_v4_public_development_v2_score_bearing_execution_allowed(
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    """Fail closed because question-only grouping cannot prove semantic families."""

    checked = BfclV4LoadedPublicDevelopmentV2.model_validate(loaded, strict=True)
    if (
        not checked.independent_semantic_family_disjointness_attested
        or not checked.score_bearing_execution_allowed
    ):
        raise BfclV4PublicDevelopmentV2ExecutionError(
            "BFCL v2 score-bearing execution is forbidden: independent semantic "
            "family disjointness is not proven"
        )
    raise BfclV4PublicDevelopmentV2ExecutionError(
        "BFCL v2 score-bearing execution is not implemented"
    )


__all__ = [
    "BfclV4PublicDevelopmentV2Error",
    "BfclV4PublicDevelopmentV2ExecutionError",
    "assert_bfcl_v4_public_development_v2_score_bearing_execution_allowed",
    "load_bfcl_v4_public_development_v2",
]
