"""Provider-free loading and rejection audit for the tau3 public pilot.

This module deliberately has no model-provider dependency.  It reads immutable
Git objects, verifies the pinned source bundle, recomputes the neutral draw,
and returns a rejected receipt.  No function in this module can turn that
receipt into a score-bearing task plan.
"""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

from spiral_harness.benchmark.tau3_public_pilot_contracts import (
    TAU3_GROUPING_AUTHORITY,
    TAU3_REJECTED_ROSTER_RECEIPT,
    TAU3_SOURCE_JACCARD_SCREEN_THRESHOLD,
    TAU3_TEMPLATE_JACCARD_SCREEN_THRESHOLD,
    TAU3_UPSTREAM_COMMIT,
    TAU3_UPSTREAM_RELEASE,
    Tau3BenchmarkDomain,
    Tau3BlindGroupingAuditBundle,
    Tau3CandidateTaskContract,
    Tau3LoadedRejectedPilot,
    Tau3NearDuplicateAudit,
    Tau3NearDuplicatePair,
    Tau3TrustedGroupingRecord,
    Tau3TrustedRosterEntry,
    tau3_selection_sha256,
)
from spiral_harness.benchmark.tau3_public_pilot_identities import (
    TAU3_CRITICAL_FILE_IDENTITIES,
    Tau3FileIdentity,
)
from spiral_harness.core.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)

_TASK_PATHS = {
    Tau3BenchmarkDomain.AIRLINE: "data/tau2/domains/airline/tasks.json",
    Tau3BenchmarkDomain.RETAIL: "data/tau2/domains/retail/tasks.json",
    Tau3BenchmarkDomain.TELECOM: "data/tau2/domains/telecom/tasks.json",
}
_SPLIT_PATHS = {
    Tau3BenchmarkDomain.AIRLINE: "data/tau2/domains/airline/split_tasks.json",
    Tau3BenchmarkDomain.RETAIL: "data/tau2/domains/retail/split_tasks.json",
    Tau3BenchmarkDomain.TELECOM: "data/tau2/domains/telecom/split_tasks.json",
}
_BASE_COUNTS = {
    Tau3BenchmarkDomain.AIRLINE: 50,
    Tau3BenchmarkDomain.RETAIL: 114,
    Tau3BenchmarkDomain.TELECOM: 114,
}
_TELECOM_PREFIXES = {
    "telecom-mobile-data-base": "[mobile_data_issue]",
    "telecom-service-base": "[service_issue]",
    "telecom-mms-base": "[mms_issue]",
}
_REQUIRED_ROW_KEYS = {
    Tau3BenchmarkDomain.AIRLINE: {
        "id",
        "description",
        "user_scenario",
        "initial_state",
        "evaluation_criteria",
        "annotations",
    },
    Tau3BenchmarkDomain.RETAIL: {
        "id",
        "description",
        "user_scenario",
        "initial_state",
        "evaluation_criteria",
    },
    Tau3BenchmarkDomain.TELECOM: {
        "id",
        "description",
        "user_scenario",
        "ticket",
        "initial_state",
        "evaluation_criteria",
    },
}
_OPTIONAL_ROW_KEYS = {
    Tau3BenchmarkDomain.AIRLINE: set(),
    Tau3BenchmarkDomain.RETAIL: {"issues"},
    Tau3BenchmarkDomain.TELECOM: set(),
}

_COMMON_STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "have",
        "has",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "they",
        "this",
        "to",
        "up",
        "want",
        "wants",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
        "don't",
        "doesn't",
        "can't",
        "won't",
        "you're",
        "you've",
        "you'll",
        "it's",
        "i'm",
        "i've",
        "isn't",
    )
)
_SOURCE_STOPWORDS = _COMMON_STOPWORDS | frozenset(
    (
        "address",
        "code",
        "confirmation",
        "customer",
        "email",
        "id",
        "live",
        "living",
        "name",
        "number",
        "phone",
        "reservation",
        "user",
        "zip",
    )
)
_EMAIL = re.compile(r"\b([a-z0-9._%+\-]+)@[a-z0-9.\-]+\.[a-z]{2,}\b", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


class Tau3PublicPilotError(RuntimeError):
    """A pinned source, rejection, or authority boundary failed closed."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_surrogates(value: Any) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON strings must not contain surrogate code points")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_surrogates(key)
            _reject_surrogates(item)
    elif isinstance(value, list):
        for item in value:
            _reject_surrogates(item)


def _strict_json(content: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate object keys")
            output[key] = value
        return output

    try:
        text = content.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Tau3PublicPilotError(f"{label} is not strict UTF-8 JSON") from error
    try:
        _reject_surrogates(value)
        canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise Tau3PublicPilotError(f"{label} violates the canonical value contract") from error
    return value


def _git(checkout: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ("git", "-C", str(checkout), *arguments),
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise Tau3PublicPilotError("pinned tau3 Git operation failed") from error
    return result.stdout


def _verify_checkout(checkout: str | Path) -> Path:
    resolved = Path(checkout).expanduser().resolve()
    if not resolved.is_dir():
        raise Tau3PublicPilotError("tau3 checkout is not a directory")
    try:
        top_level = Path(_git(resolved, "rev-parse", "--show-toplevel").decode().strip()).resolve()
        head = _git(resolved, "rev-parse", "HEAD").decode().strip()
        release = _git(resolved, "describe", "--tags", "--exact-match", "HEAD").decode().strip()
    except UnicodeDecodeError as error:
        raise Tau3PublicPilotError("tau3 Git metadata is not UTF-8") from error
    if top_level != resolved:
        raise Tau3PublicPilotError("tau3 checkout argument must be the repository root")
    if head != TAU3_UPSTREAM_COMMIT or release != TAU3_UPSTREAM_RELEASE:
        raise Tau3PublicPilotError("tau3 checkout differs from v1.0.1 at the pinned commit")
    return resolved


def _read_pinned_blob(checkout: Path, git_path: str) -> bytes:
    return _git(checkout, "show", f"{TAU3_UPSTREAM_COMMIT}:{git_path}")


def _verify_blob(content: bytes, identity: Tau3FileIdentity) -> bytes:
    if len(content) != identity.size or sha256_bytes(content) != identity.sha256:
        raise Tau3PublicPilotError(f"pinned tau3 source differs: {identity.git_path}")
    return content


def _load_verified_blobs(checkout: str | Path) -> dict[str, bytes]:
    resolved = _verify_checkout(checkout)
    return {
        identity.git_path: _verify_blob(
            _read_pinned_blob(resolved, identity.git_path),
            identity,
        )
        for identity in TAU3_CRITICAL_FILE_IDENTITIES
    }


def _task_rows(value: Any, domain: Tau3BenchmarkDomain) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise Tau3PublicPilotError(f"{domain.value} tasks are not a JSON list")
    output: dict[str, dict[str, Any]] = {}
    for row in value:
        if (
            not isinstance(row, dict)
            or not _REQUIRED_ROW_KEYS[domain] <= set(row)
            or not set(row) <= _REQUIRED_ROW_KEYS[domain] | _OPTIONAL_ROW_KEYS[domain]
        ):
            raise Tau3PublicPilotError(f"{domain.value} task row schema changed")
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id or task_id != task_id.strip():
            raise Tau3PublicPilotError(f"{domain.value} task ID is not canonical")
        if task_id in output:
            raise Tau3PublicPilotError(f"{domain.value} repeats task ID {task_id}")
        output[task_id] = row
    return output


def _base_ids(
    value: Any,
    domain: Tau3BenchmarkDomain,
    available_ids: frozenset[str],
) -> tuple[str, ...]:
    if not isinstance(value, dict) or not isinstance(value.get("base"), list):
        raise Tau3PublicPilotError(f"{domain.value} split lacks a base list")
    raw_ids = value["base"]
    if any(not isinstance(item, str) or not item or item != item.strip() for item in raw_ids):
        raise Tau3PublicPilotError(f"{domain.value} base split has a non-canonical ID")
    ids = tuple(raw_ids)
    if len(ids) != _BASE_COUNTS[domain] or len(set(ids)) != len(ids):
        raise Tau3PublicPilotError(f"{domain.value} base split cardinality changed")
    if not set(ids) <= available_ids:
        raise Tau3PublicPilotError(f"{domain.value} base split cites an absent task")
    return ids


def _selected_ids_for_domain(
    domain: Tau3BenchmarkDomain, base_ids: tuple[str, ...]
) -> tuple[str, ...]:
    ordered = tuple(sorted(base_ids, key=lambda item: (tau3_selection_sha256(domain, item), item)))
    if domain != Tau3BenchmarkDomain.TELECOM:
        return ordered[:6]
    selected: list[str] = []
    for prefix in _TELECOM_PREFIXES.values():
        pool = tuple(item for item in ordered if item.startswith(prefix))
        if not pool:
            raise Tau3PublicPilotError(f"telecom base split lacks prefix {prefix}")
        selected.append(pool[0])
    return tuple(selected)


def _source_projection(domain: Tau3BenchmarkDomain, row: Mapping[str, Any]) -> dict[str, Any]:
    projection = {
        "id": row["id"],
        "description": row["description"],
        "user_scenario": row["user_scenario"],
    }
    if domain == Tau3BenchmarkDomain.TELECOM:
        projection["ticket"] = row["ticket"]
    return projection


def _scenario_parts(row: Mapping[str, Any]) -> tuple[str, str, str, str | None]:
    description = row.get("description")
    scenario = row.get("user_scenario")
    if not isinstance(description, dict) or not isinstance(scenario, dict):
        raise Tau3PublicPilotError("tau3 task description or user scenario is not an object")
    instructions = scenario.get("instructions")
    if not isinstance(instructions, dict):
        raise Tau3PublicPilotError("tau3 user scenario lacks an instructions object")
    purpose = description.get("purpose")
    if purpose is not None and not isinstance(purpose, str):
        raise Tau3PublicPilotError("tau3 task purpose is not text or null")
    values = tuple(instructions.get(key) for key in ("reason_for_call", "task_instructions"))
    if any(not isinstance(item, str) for item in values):
        raise Tau3PublicPilotError("tau3 template fields are not text")
    known_info = instructions.get("known_info") or ""
    if not isinstance(known_info, str):
        raise Tau3PublicPilotError("tau3 known-info field is not text or null")
    reason, task_instructions = values
    return reason, task_instructions, known_info, purpose


def _template_text(row: Mapping[str, Any]) -> str:
    reason, task_instructions, _, purpose = _scenario_parts(row)
    return "\n".join((purpose or "", reason, task_instructions))


def _source_text(row: Mapping[str, Any]) -> str:
    return _scenario_parts(row)[2]


def _normalized_tokens(text: str, *, source: bool) -> tuple[str, ...]:
    normalized = (
        unicodedata.normalize("NFKC", text)
        .casefold()
        .replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    )
    normalized = _EMAIL.sub(lambda match: f" {match.group(1)} email ", normalized)
    normalized = _DIGITS.sub(" number ", normalized)
    stopwords = _SOURCE_STOPWORDS if source else _COMMON_STOPWORDS | {"email", "number"}
    return tuple(sorted({token for token in _TOKEN.findall(normalized) if token not in stopwords}))


def _verify_row(entry: Tau3TrustedRosterEntry, row: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    domain = entry.benchmark_domain
    projection = _source_projection(domain, row)
    template = _template_text(row)
    source = _source_text(row)
    template_tokens = _normalized_tokens(template, source=False)
    source_tokens = _normalized_tokens(source, source=True)
    actual = (
        len(canonical_json_bytes(row)),
        canonical_sha256(row),
        len(canonical_json_bytes(projection)),
        canonical_sha256(projection),
        len(template.encode("utf-8")),
        sha256_bytes(template.encode("utf-8")),
        len(template_tokens),
        canonical_sha256(template_tokens),
        len(source.encode("utf-8")),
        sha256_bytes(source.encode("utf-8")),
        len(source_tokens),
        canonical_sha256(source_tokens),
    )
    expected = (
        entry.row_size,
        entry.row_sha256,
        entry.source_projection_size,
        entry.source_projection_sha256,
        entry.template_text_size,
        entry.template_text_sha256,
        entry.template_token_count,
        entry.template_token_sha256,
        entry.source_text_size,
        entry.source_text_sha256,
        entry.source_token_count,
        entry.source_token_sha256,
    )
    if actual != expected:
        raise Tau3PublicPilotError(f"selected tau3 row identity changed: {entry.opaque_task_id}")
    return set(template_tokens), set(source_tokens)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _near_duplicate_audit(
    token_sets: Mapping[str, tuple[set[str], set[str]]],
) -> Tau3NearDuplicateAudit:
    entries = TAU3_REJECTED_ROSTER_RECEIPT.roster
    pairs: list[Tau3NearDuplicatePair] = []
    for left, right in combinations(entries, 2):
        left_template, left_source = token_sets[left.opaque_task_id]
        right_template, right_source = token_sets[right.opaque_task_id]
        pairs.append(
            Tau3NearDuplicatePair(
                left_opaque_task_id=left.opaque_task_id,
                right_opaque_task_id=right.opaque_task_id,
                cross_partition=left.partition != right.partition,
                same_source_cluster=left.source_cluster_id == right.source_cluster_id,
                template_jaccard=_jaccard(left_template, right_template),
                source_jaccard=_jaccard(left_source, right_source),
                exact_template_collision=left_template == right_template,
                exact_source_collision=left_source == right_source,
            )
        )
    automatic_violations = sum(
        item.cross_partition
        and (
            item.exact_template_collision
            or item.exact_source_collision
            or item.template_jaccard >= TAU3_TEMPLATE_JACCARD_SCREEN_THRESHOLD
            or item.source_jaccard >= TAU3_SOURCE_JACCARD_SCREEN_THRESHOLD
        )
        for item in pairs
    )
    return Tau3NearDuplicateAudit(
        pairs=tuple(pairs),
        automatic_cross_partition_violations=automatic_violations,
        automatic_screen_passed=automatic_violations == 0,
        independent_semantic_conflicts=TAU3_REJECTED_ROSTER_RECEIPT.semantic_conflicts,
    )


def _load_task_sources(
    blobs: Mapping[str, bytes],
) -> tuple[
    dict[Tau3BenchmarkDomain, dict[str, dict[str, Any]]],
    dict[Tau3BenchmarkDomain, tuple[str, ...]],
]:
    rows_by_domain: dict[Tau3BenchmarkDomain, dict[str, dict[str, Any]]] = {}
    base_by_domain: dict[Tau3BenchmarkDomain, tuple[str, ...]] = {}
    for domain in Tau3BenchmarkDomain:
        rows = _task_rows(_strict_json(blobs[_TASK_PATHS[domain]], _TASK_PATHS[domain]), domain)
        base = _base_ids(
            _strict_json(blobs[_SPLIT_PATHS[domain]], _SPLIT_PATHS[domain]),
            domain,
            frozenset(rows),
        )
        rows_by_domain[domain] = rows
        base_by_domain[domain] = base
    return rows_by_domain, base_by_domain


def _verify_neutral_draw(
    base_by_domain: Mapping[Tau3BenchmarkDomain, tuple[str, ...]],
) -> None:
    for domain in Tau3BenchmarkDomain:
        selected = _selected_ids_for_domain(domain, base_by_domain[domain])
        receipt_entries = tuple(
            sorted(
                (
                    item
                    for item in TAU3_REJECTED_ROSTER_RECEIPT.roster
                    if item.benchmark_domain == domain
                ),
                key=lambda item: (item.selection_stratum, item.selection_rank),
            )
        )
        if domain == Tau3BenchmarkDomain.TELECOM:
            by_stratum = {item.selection_stratum: item for item in receipt_entries}
            expected = tuple(by_stratum[stratum].upstream_task_id for stratum in _TELECOM_PREFIXES)
        else:
            expected = tuple(item.upstream_task_id for item in receipt_entries)
        if expected != selected:
            raise Tau3PublicPilotError(f"{domain.value} receipt differs from the neutral draw")
        for entry in receipt_entries:
            if entry.selection_pool_size != (
                len(base_by_domain[domain])
                if domain != Tau3BenchmarkDomain.TELECOM
                else sum(
                    task_id.startswith(_TELECOM_PREFIXES[entry.selection_stratum])
                    for task_id in base_by_domain[domain]
                )
            ):
                raise Tau3PublicPilotError("selection pool size differs from the pinned base split")


def load_tau3_rejected_public_pilot(checkout: str | Path) -> Tau3LoadedRejectedPilot:
    """Verify the pinned sources and return the non-executable rejection receipt."""

    blobs = _load_verified_blobs(checkout)
    rows_by_domain, base_by_domain = _load_task_sources(blobs)
    _verify_neutral_draw(base_by_domain)
    token_sets: dict[str, tuple[set[str], set[str]]] = {}
    for entry in TAU3_REJECTED_ROSTER_RECEIPT.roster:
        row = rows_by_domain[entry.benchmark_domain][entry.upstream_task_id]
        token_sets[entry.opaque_task_id] = _verify_row(entry, row)
    return Tau3LoadedRejectedPilot(
        receipt=TAU3_REJECTED_ROSTER_RECEIPT,
        near_duplicate_audit=_near_duplicate_audit(token_sets),
        authority=TAU3_GROUPING_AUTHORITY,
    )


def _structured_prefix(domain: Tau3BenchmarkDomain, task_id: str) -> str | None:
    if domain != Tau3BenchmarkDomain.TELECOM:
        return None
    closing = task_id.find("]")
    if closing <= 0:
        raise Tau3PublicPilotError("telecom task ID lacks its structured prefix")
    prefix = task_id[: closing + 1]
    if prefix not in set(_TELECOM_PREFIXES.values()):
        raise Tau3PublicPilotError("telecom task ID has an unknown structured prefix")
    return prefix


def _grouping_record(
    domain: Tau3BenchmarkDomain,
    row: Mapping[str, Any],
) -> Tau3TrustedGroupingRecord:
    reason, task_instructions, known_info, purpose = _scenario_parts(row)
    criteria = row.get("evaluation_criteria")
    if not isinstance(criteria, dict):
        raise Tau3PublicPilotError("base task lacks evaluation criteria for trusted grouping")
    actions = criteria.get("actions") or []
    reward_basis = criteria.get("reward_basis") or []
    if not isinstance(actions, list) or not isinstance(reward_basis, list):
        raise Tau3PublicPilotError("trusted grouping action or reward basis is not a list")
    action_signature: list[str] = []
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            raise Tau3PublicPilotError("trusted grouping action schema changed")
        requestor = action.get("requestor") or "unspecified"
        if not isinstance(requestor, str):
            raise Tau3PublicPilotError("trusted grouping action requestor is not text")
        action_signature.append(f"{requestor}:{action['name']}")
    if any(not isinstance(item, str) or not item for item in reward_basis):
        raise Tau3PublicPilotError("trusted grouping reward basis is invalid")
    projection_sha256 = canonical_sha256(_source_projection(domain, row))
    return Tau3TrustedGroupingRecord(
        audit_record_id=canonical_sha256(
            {
                "domain": "spiral-tau3-blind-source-family-audit/v1",
                "benchmark_domain": domain.value,
                "upstream_task_id": row["id"],
                "source_projection_sha256": projection_sha256,
            }
        ),
        benchmark_domain=domain,
        upstream_task_id=row["id"],
        source_projection_sha256=projection_sha256,
        purpose=purpose,
        reason_for_call=reason,
        task_instructions=task_instructions,
        known_info=known_info,
        structured_id_prefix=_structured_prefix(domain, row["id"]),
        ordered_action_signature=tuple(action_signature),
        reward_basis=tuple(reward_basis),
    )


def load_tau3_blind_grouping_audit_bundle(
    checkout: str | Path,
) -> Tau3BlindGroupingAuditBundle:
    """Build all 278 trusted inputs required for independent family annotation."""

    blobs = _load_verified_blobs(checkout)
    rows_by_domain, base_by_domain = _load_task_sources(blobs)
    records = tuple(
        _grouping_record(domain, rows_by_domain[domain][task_id])
        for domain in Tau3BenchmarkDomain
        for task_id in sorted(base_by_domain[domain])
    )
    return Tau3BlindGroupingAuditBundle(
        authority=TAU3_GROUPING_AUTHORITY,
        records=records,
    )


def release_tau3_candidate_search_contracts(
    loaded: Tau3LoadedRejectedPilot,
) -> tuple[Tau3CandidateTaskContract, ...]:
    """Fail closed: a rejected roster can never release search contracts."""

    checked = Tau3LoadedRejectedPilot.model_validate(loaded, strict=True)
    if not checked.receipt.roster_admissible or not checked.authority.execution_release_allowed:
        raise Tau3PublicPilotError(
            "tau3 roster rejected: independent source-family grouping is incomplete"
        )
    raise Tau3PublicPilotError("tau3 executable roster release is not implemented")


def assert_tau3_score_bearing_execution_allowed(loaded: Tau3LoadedRejectedPilot) -> None:
    """Fail closed at score-bearing runner boundaries."""

    release_tau3_candidate_search_contracts(loaded)


__all__ = [name for name in globals() if name.startswith("Tau3") or name.startswith("load_tau3")]
