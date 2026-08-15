"""Fail-closed contracts for the tau-cubed v1.0.1 public pilot prerequisite.

The deterministic row selection recorded here was rejected before any model
call.  It is retained as an auditable negative receipt, never as an executable
benchmark plan.  A future roster must first group the complete base suite into
source families under the independent authority contract below.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from spiral_harness.benchmark.tau3_public_pilot_identities import (
    TAU3_CRITICAL_FILE_BUNDLE_SHA256,
    TAU3_CRITICAL_FILE_IDENTITIES,
    TAU3_DOMAIN_POLICY_BUNDLE_SHA256,
    TAU3_DOMAIN_POLICY_PATHS,
    TAU3_DOMAIN_TOOL_BUNDLE_SHA256,
    TAU3_DOMAIN_TOOL_PATHS,
    TAU3_PILOT_ROW_IDENTITIES,
)
from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

TAU3_UPSTREAM_REPOSITORY = "https://github.com/sierra-research/tau2-bench.git"
TAU3_UPSTREAM_RELEASE = "v1.0.1"
TAU3_UPSTREAM_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU3_SELECTION_SALT = "tau3:v1.0.1:fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU3_TEMPLATE_JACCARD_SCREEN_THRESHOLD = 0.35
TAU3_SOURCE_JACCARD_SCREEN_THRESHOLD = 0.40


class Tau3BenchmarkDomain(StrEnum):
    """Pinned text-mode domains in the public prerequisite."""

    AIRLINE = "airline"
    RETAIL = "retail"
    TELECOM = "telecom"


class Tau3PilotPartition(StrEnum):
    """Intended partitions of the rejected deterministic draw."""

    FIT = "fit"
    GATE = "gate"
    HOLDOUT = "holdout"


class Tau3SemanticConflictCode(StrEnum):
    """Pre-result reasons the deterministic draw was rejected."""

    DELAY_COMPENSATION = "cross-partition-delayed-flight-compensation"
    ADDRESS_CHANGE = "cross-partition-address-change"


def tau3_selection_sha256(domain: str, task_id: str) -> str:
    """Hash the exact public salt/domain/task-ID byte string.

    The separators are single NUL bytes.  Domain and task IDs are their
    canonical JSON string values encoded directly as UTF-8, without quotes,
    whitespace, or a trailing newline.
    """

    if domain not in {item.value for item in Tau3BenchmarkDomain}:
        raise ValueError("tau3 selection domain is not canonical")
    if not isinstance(task_id, str) or not task_id or task_id != task_id.strip():
        raise ValueError("tau3 selection task ID must be a canonical non-empty string")
    payload = b"\x00".join(
        (TAU3_SELECTION_SALT.encode("utf-8"), domain.encode("utf-8"), task_id.encode("utf-8"))
    )
    return sha256_bytes(payload)


def tau3_critical_file_bundle_sha256() -> str:
    """Recompute the ordered 75-file identity-bundle commitment."""

    return canonical_sha256(
        tuple(
            {"git_path": item.git_path, "size": item.size, "sha256": item.sha256}
            for item in TAU3_CRITICAL_FILE_IDENTITIES
        )
    )


def tau3_source_subset_bundle_sha256(paths: tuple[str, ...]) -> str:
    """Bind an ordered policy/tool surface to the pinned file identities."""

    if not paths or len(paths) != len(set(paths)):
        raise ValueError("source subset paths must be a non-empty ordered set")
    identities = {item.git_path: item for item in TAU3_CRITICAL_FILE_IDENTITIES}
    try:
        selected = tuple(identities[path] for path in paths)
    except KeyError as error:
        raise ValueError("source subset path is absent from the critical bundle") from error
    return canonical_sha256(
        tuple(
            {"git_path": item.git_path, "size": item.size, "sha256": item.sha256}
            for item in selected
        )
    )


class Tau3TrustedRosterEntry(ImmutableModel):
    """Trusted-only coordinates for one row in the rejected receipt."""

    schema_version: Literal["1"] = "1"
    ordinal: Annotated[int, Field(ge=0, lt=15, strict=True)]
    opaque_task_id: NonEmptyStr
    benchmark_domain: Tau3BenchmarkDomain
    upstream_task_id: NonEmptyStr
    partition: Tau3PilotPartition
    selection_stratum: NonEmptyStr
    selection_pool_size: Annotated[int, Field(gt=0, strict=True)]
    selection_rank: Annotated[int, Field(ge=0, strict=True)]
    source_cluster_id: NonEmptyStr
    selection_sha256: Sha256
    row_size: Annotated[int, Field(gt=0, strict=True)]
    row_sha256: Sha256
    source_projection_size: Annotated[int, Field(gt=0, strict=True)]
    source_projection_sha256: Sha256
    template_text_size: Annotated[int, Field(gt=0, strict=True)]
    template_text_sha256: Sha256
    template_token_count: Annotated[int, Field(gt=0, strict=True)]
    template_token_sha256: Sha256
    source_text_size: Annotated[int, Field(gt=0, strict=True)]
    source_text_sha256: Sha256
    source_token_count: Annotated[int, Field(gt=0, strict=True)]
    source_token_sha256: Sha256
    candidate_visible: Literal[False] = False

    @model_validator(mode="after")
    def _bind_identity(self) -> Self:
        try:
            expected = TAU3_PILOT_ROW_IDENTITIES[self.opaque_task_id]
        except KeyError as error:
            raise ValueError("opaque task ID is absent from the rejected receipt") from error
        actual = (
            self.benchmark_domain.value,
            self.upstream_task_id,
            self.partition.value,
            self.selection_stratum,
            self.selection_pool_size,
            self.selection_rank,
            self.source_cluster_id,
            self.selection_sha256,
            self.row_size,
            self.row_sha256,
            self.source_projection_size,
            self.source_projection_sha256,
            self.template_text_size,
            self.template_text_sha256,
            self.template_token_count,
            self.template_token_sha256,
            self.source_text_size,
            self.source_text_sha256,
            self.source_token_count,
            self.source_token_sha256,
        )
        if actual != expected:
            raise ValueError("trusted roster entry differs from its pinned row identity")
        if self.selection_sha256 != tau3_selection_sha256(
            self.benchmark_domain.value, self.upstream_task_id
        ):
            raise ValueError("selection digest differs from the frozen byte formula")
        return self


class Tau3SemanticConflict(ImmutableModel):
    """One independently spotted conflict that blocks partition release."""

    code: Tau3SemanticConflictCode
    opaque_task_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=2)]
    partitions: Annotated[tuple[Tau3PilotPartition, ...], Field(min_length=2)]
    source: Literal["pre-model-read-only-semantic-audit"] = "pre-model-read-only-semantic-audit"
    requires_same_source_family_before_partition: Literal[True] = True
    model_results_used: Literal[False] = False

    @model_validator(mode="after")
    def _close_conflict(self) -> Self:
        if len(self.opaque_task_ids) != len(self.partitions):
            raise ValueError("semantic conflict members and partitions differ in length")
        if len(set(self.opaque_task_ids)) != len(self.opaque_task_ids):
            raise ValueError("semantic conflict repeats a task")
        if len(set(self.partitions)) < 2:
            raise ValueError("semantic conflict must cross intended partitions")
        return self


class Tau3RejectedRosterReceipt(ImmutableModel):
    """Immutable evidence that this deterministic draw must not execute."""

    schema_version: Literal["1"] = "1"
    suite_id: Literal["tau3-public-dry-run@v1.0.1"] = "tau3-public-dry-run@v1.0.1"
    upstream_repository: Literal[TAU3_UPSTREAM_REPOSITORY] = TAU3_UPSTREAM_REPOSITORY
    upstream_release: Literal[TAU3_UPSTREAM_RELEASE] = TAU3_UPSTREAM_RELEASE
    upstream_commit: Literal[TAU3_UPSTREAM_COMMIT] = TAU3_UPSTREAM_COMMIT
    selection_salt: Literal[TAU3_SELECTION_SALT] = TAU3_SELECTION_SALT
    selection_rule: Literal[
        "sha256(utf8(salt)||nul||utf8(domain)||nul||utf8(canonical-task-id))"
    ] = "sha256(utf8(salt)||nul||utf8(domain)||nul||utf8(canonical-task-id))"
    roster: Annotated[tuple[Tau3TrustedRosterEntry, ...], Field(min_length=15, max_length=15)]
    critical_file_count: Literal[75] = 75
    critical_file_bundle_sha256: Literal[TAU3_CRITICAL_FILE_BUNDLE_SHA256] = (
        TAU3_CRITICAL_FILE_BUNDLE_SHA256
    )
    semantic_conflicts: Annotated[
        tuple[Tau3SemanticConflict, ...], Field(min_length=2, max_length=2)
    ]
    evidence_scope: Literal["public-development-rejected-roster-receipt"] = (
        "public-development-rejected-roster-receipt"
    )
    upstream_lineage_attested: Literal[False] = False
    automated_similarity_is_sufficient_lineage: Literal[False] = False
    roster_admissible: Literal[False] = False
    confirmatory_eligible: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    candidate_contract_release_allowed: Literal[False] = False
    self_evolution_evidence: Literal[False] = False
    search_seeds_u64: Annotated[tuple[int, ...], Field(max_length=0)] = ()
    trajectories_frozen: Literal[False] = False
    budget_topology_frozen: Literal[False] = False
    model_invoked: Literal[False] = False

    @model_validator(mode="after")
    def _close_rejected_receipt(self) -> Self:
        if tuple(item.ordinal for item in self.roster) != tuple(range(15)):
            raise ValueError("rejected roster ordinals must be 0..14")
        if len({item.opaque_task_id for item in self.roster}) != 15:
            raise ValueError("rejected roster opaque IDs must be unique")
        if len({(item.benchmark_domain, item.upstream_task_id) for item in self.roster}) != 15:
            raise ValueError("rejected roster upstream coordinates must be unique")
        if Counter(item.partition for item in self.roster) != {
            Tau3PilotPartition.FIT: 4,
            Tau3PilotPartition.GATE: 4,
            Tau3PilotPartition.HOLDOUT: 7,
        }:
            raise ValueError("rejected roster must retain the intended 4/4/7 shape")
        if Counter(item.benchmark_domain for item in self.roster) != {
            Tau3BenchmarkDomain.AIRLINE: 6,
            Tau3BenchmarkDomain.RETAIL: 6,
            Tau3BenchmarkDomain.TELECOM: 3,
        }:
            raise ValueError("rejected roster must retain the intended domain shape")
        for domain in (Tau3BenchmarkDomain.AIRLINE, Tau3BenchmarkDomain.RETAIL):
            ranks = tuple(
                item.selection_rank for item in self.roster if item.benchmark_domain == domain
            )
            if ranks != tuple(range(6)):
                raise ValueError("global domain draws must retain ranks 0..5")
        telecom = tuple(
            item for item in self.roster if item.benchmark_domain == Tau3BenchmarkDomain.TELECOM
        )
        if {item.selection_rank for item in telecom} != {0}:
            raise ValueError("each telecom prefix representative must be rank zero")
        if len({item.source_cluster_id for item in telecom}) != 1:
            raise ValueError("telecom representatives must share one source cluster")
        conflict_members = {
            task_id for item in self.semantic_conflicts for task_id in item.opaque_task_ids
        }
        if not conflict_members <= {item.opaque_task_id for item in self.roster}:
            raise ValueError("semantic conflict cites a task outside the receipt")
        if self.critical_file_bundle_sha256 != tau3_critical_file_bundle_sha256():
            raise ValueError("critical source bundle differs from the pinned identities")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class Tau3CandidateTaskContract(ImmutableModel):
    """The complete future candidate-visible task envelope.

    It intentionally contains no partition, source group, upstream ID, user
    scenario, initial state, evaluation criteria, gold action, or grader field.
    The current rejected receipt is forbidden from releasing even this view.
    """

    schema_version: Literal["1"] = "1"
    opaque_task_id: NonEmptyStr
    benchmark_domain: Tau3BenchmarkDomain
    policy_surface_sha256: Sha256
    tool_surface_sha256: Sha256
    user_messages_supplied_dynamically: Literal[True] = True

    @model_validator(mode="after")
    def _bind_candidate_surfaces(self) -> Self:
        domain = self.benchmark_domain.value
        expected_policy = tau3_source_subset_bundle_sha256(TAU3_DOMAIN_POLICY_PATHS[domain])
        expected_tools = tau3_source_subset_bundle_sha256(TAU3_DOMAIN_TOOL_PATHS[domain])
        if (
            self.policy_surface_sha256 != TAU3_DOMAIN_POLICY_BUNDLE_SHA256[domain]
            or self.policy_surface_sha256 != expected_policy
        ):
            raise ValueError("candidate policy surface differs from the pinned domain")
        if (
            self.tool_surface_sha256 != TAU3_DOMAIN_TOOL_BUNDLE_SHA256[domain]
            or self.tool_surface_sha256 != expected_tools
        ):
            raise ValueError("candidate tool surface differs from the pinned domain")
        return self


class Tau3NearDuplicatePair(ImmutableModel):
    """Trusted pairwise similarity record for the rejected 15-row draw."""

    left_opaque_task_id: NonEmptyStr
    right_opaque_task_id: NonEmptyStr
    cross_partition: bool
    same_source_cluster: bool
    template_jaccard: Annotated[float, Field(ge=0.0, le=1.0)]
    source_jaccard: Annotated[float, Field(ge=0.0, le=1.0)]
    exact_template_collision: bool
    exact_source_collision: bool


class Tau3NearDuplicateAudit(ImmutableModel):
    """Automatic screen plus the mandatory independent semantic decision."""

    schema_version: Literal["1"] = "1"
    pairs: Annotated[tuple[Tau3NearDuplicatePair, ...], Field(min_length=105, max_length=105)]
    template_threshold: Literal[TAU3_TEMPLATE_JACCARD_SCREEN_THRESHOLD] = (
        TAU3_TEMPLATE_JACCARD_SCREEN_THRESHOLD
    )
    source_threshold: Literal[TAU3_SOURCE_JACCARD_SCREEN_THRESHOLD] = (
        TAU3_SOURCE_JACCARD_SCREEN_THRESHOLD
    )
    automatic_cross_partition_violations: Annotated[int, Field(ge=0, strict=True)]
    automatic_screen_passed: bool
    independent_semantic_conflicts: Annotated[
        tuple[Tau3SemanticConflict, ...], Field(min_length=2, max_length=2)
    ]
    independent_semantic_audit_passed: Literal[False] = False
    overall_passed: Literal[False] = False
    model_results_used: Literal[False] = False

    @model_validator(mode="after")
    def _close_audit(self) -> Self:
        expected_automatic = sum(
            item.cross_partition
            and (
                item.exact_template_collision
                or item.exact_source_collision
                or item.template_jaccard >= self.template_threshold
                or item.source_jaccard >= self.source_threshold
            )
            for item in self.pairs
        )
        if self.automatic_cross_partition_violations != expected_automatic:
            raise ValueError("automatic near-duplicate violation count is inconsistent")
        if self.automatic_screen_passed != (expected_automatic == 0):
            raise ValueError("automatic near-duplicate pass flag is inconsistent")
        return self


class Tau3BaselineBoundary(ImmutableModel):
    """Arm meanings that remain fixed even while execution is blocked."""

    schema_version: Literal["1"] = "1"
    primary_bare_agent_baseline: Literal["NATIVE-MIN"] = "NATIVE-MIN"
    native_min_definition: Literal[
        "fixed-upstream-llm-agent-policy-orchestrator-and-domain-tools"
    ] = "fixed-upstream-llm-agent-policy-orchestrator-and-domain-tools"
    pure_role: Literal["secondary-structurally-weak-provider-minimal-reference"] = (
        "secondary-structurally-weak-provider-minimal-reference"
    )
    pure_has_domain_tools: Literal[False] = False
    pure_at_b_status: Literal["blocked-undefined"] = "blocked-undefined"
    pure_at_b_blocker: Literal["no-target-free-trajectory-to-policy-aggregator"] = (
        "no-target-free-trajectory-to-policy-aggregator"
    )
    native_min_may_be_renamed_pure: Literal[False] = False
    current_score_bearing_arms_executable: Literal[False] = False


class Tau3GroupingAuthorityContract(ImmutableModel):
    """Authority needed before another roster may be partitioned or selected."""

    schema_version: Literal["1"] = "1"
    status: Literal["blocked-awaiting-independent-double-annotation"] = (
        "blocked-awaiting-independent-double-annotation"
    )
    full_base_roster_size: Literal[278] = 278
    independent_annotators_required: Literal[2] = 2
    independent_adjudicators_required: Literal[1] = 1
    full_roster_annotated_before_partition: Literal[True] = True
    source_family_before_partition: Literal[True] = True
    every_source_family_confined_to_one_partition: Literal[True] = True
    within_group_selection_after_partition: Literal[True] = True
    within_group_selection_rule: Literal[
        "frozen-nul-separated-salt-domain-canonical-task-id-sha256-argmin"
    ] = "frozen-nul-separated-salt-domain-canonical-task-id-sha256-argmin"
    annotation_blind_to_model_outputs: Literal[True] = True
    annotation_blind_to_harness_outputs: Literal[True] = True
    annotation_blind_to_candidate_artifacts: Literal[True] = True
    candidate_receives_group_labels: Literal[False] = False
    post_partition_task_reassignment_allowed: Literal[False] = False
    automatic_grouping_validated: Literal[False] = False
    independent_double_annotation_complete: Literal[False] = False
    adjudication_complete: Literal[False] = False
    execution_release_allowed: Literal[False] = False


class Tau3TrustedGroupingRecord(ImmutableModel):
    """Trusted annotation input; never a candidate-facing task contract."""

    model_config = ConfigDict(**{**ImmutableModel.model_config, "str_strip_whitespace": False})

    schema_version: Literal["1"] = "1"
    audit_record_id: NonEmptyStr
    benchmark_domain: Tau3BenchmarkDomain
    upstream_task_id: NonEmptyStr
    source_projection_sha256: Sha256
    purpose: str | None
    reason_for_call: str
    task_instructions: str
    known_info: str
    structured_id_prefix: str | None
    ordered_action_signature: tuple[NonEmptyStr, ...]
    reward_basis: tuple[NonEmptyStr, ...]
    candidate_visible: Literal[False] = False
    model_or_harness_results_present: Literal[False] = False


class Tau3BlindGroupingAuditBundle(ImmutableModel):
    """Complete pre-partition authority input for independent annotation."""

    schema_version: Literal["1"] = "1"
    authority: Tau3GroupingAuthorityContract
    records: Annotated[tuple[Tau3TrustedGroupingRecord, ...], Field(min_length=278, max_length=278)]
    partition_labels_present: Literal[False] = False
    selection_ranks_present: Literal[False] = False
    candidate_visible: Literal[False] = False
    model_or_harness_results_present: Literal[False] = False

    @model_validator(mode="after")
    def _cover_base_roster_once(self) -> Self:
        coordinates = tuple((item.benchmark_domain, item.upstream_task_id) for item in self.records)
        if len(set(coordinates)) != self.authority.full_base_roster_size:
            raise ValueError("grouping audit does not cover every base task exactly once")
        return self


class Tau3LoadedRejectedPilot(ImmutableModel):
    """Provider-free load receipt with no executable task release."""

    schema_version: Literal["1"] = "1"
    receipt: Tau3RejectedRosterReceipt
    near_duplicate_audit: Tau3NearDuplicateAudit
    authority: Tau3GroupingAuthorityContract
    candidate_contracts_released: Literal[False] = False
    model_invoked: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _retain_rejection(self) -> Self:
        if self.near_duplicate_audit.independent_semantic_conflicts != (
            self.receipt.semantic_conflicts
        ):
            raise ValueError("loaded semantic audit differs from the rejected receipt")
        if self.authority.execution_release_allowed:
            raise ValueError("incomplete grouping authority cannot release execution")
        return self


def _entry(ordinal: int, opaque_task_id: str) -> Tau3TrustedRosterEntry:
    identity = TAU3_PILOT_ROW_IDENTITIES[opaque_task_id]
    return Tau3TrustedRosterEntry(
        ordinal=ordinal,
        opaque_task_id=opaque_task_id,
        benchmark_domain=Tau3BenchmarkDomain(identity.benchmark_domain),
        upstream_task_id=identity.upstream_task_id,
        partition=Tau3PilotPartition(identity.partition),
        selection_stratum=identity.selection_stratum,
        selection_pool_size=identity.selection_pool_size,
        selection_rank=identity.selection_rank,
        source_cluster_id=identity.source_cluster_id,
        selection_sha256=identity.selection_sha256,
        row_size=identity.row_size,
        row_sha256=identity.row_sha256,
        source_projection_size=identity.source_projection_size,
        source_projection_sha256=identity.source_projection_sha256,
        template_text_size=identity.template_text_size,
        template_text_sha256=identity.template_text_sha256,
        template_token_count=identity.template_token_count,
        template_token_sha256=identity.template_token_sha256,
        source_text_size=identity.source_text_size,
        source_text_sha256=identity.source_text_sha256,
        source_token_count=identity.source_token_count,
        source_token_sha256=identity.source_token_sha256,
    )


_ROSTER_ORDER = (
    "tau3p-635ce58fdabd481b",
    "tau3p-cb806d5de2884063",
    "tau3p-7f5bc4ad03a5494a",
    "tau3p-bf591c807eaa42d3",
    "tau3p-f1033b915ea74be1",
    "tau3p-7e42d1a33f114910",
    "tau3p-ce4402047ad147ff",
    "tau3p-ccddf871db01464a",
    "tau3p-1399f08cea4e4ca8",
    "tau3p-71909394dbb4431f",
    "tau3p-22852e721395489d",
    "tau3p-02b26e459dae4959",
    "tau3p-983712d857d044c4",
    "tau3p-fb510248e09f4e52",
    "tau3p-f115420413bc4901",
)
TAU3_REJECTED_ROSTER = tuple(_entry(index, task_id) for index, task_id in enumerate(_ROSTER_ORDER))

TAU3_SEMANTIC_CONFLICTS = (
    Tau3SemanticConflict(
        code=Tau3SemanticConflictCode.DELAY_COMPENSATION,
        opaque_task_ids=("tau3p-635ce58fdabd481b", "tau3p-f1033b915ea74be1"),
        partitions=(Tau3PilotPartition.FIT, Tau3PilotPartition.GATE),
    ),
    Tau3SemanticConflict(
        code=Tau3SemanticConflictCode.ADDRESS_CHANGE,
        opaque_task_ids=(
            "tau3p-bf591c807eaa42d3",
            "tau3p-ce4402047ad147ff",
            "tau3p-02b26e459dae4959",
        ),
        partitions=(
            Tau3PilotPartition.FIT,
            Tau3PilotPartition.GATE,
            Tau3PilotPartition.HOLDOUT,
        ),
    ),
)

TAU3_REJECTED_ROSTER_RECEIPT = Tau3RejectedRosterReceipt(
    roster=TAU3_REJECTED_ROSTER,
    semantic_conflicts=TAU3_SEMANTIC_CONFLICTS,
)
TAU3_BASELINE_BOUNDARY = Tau3BaselineBoundary()
TAU3_GROUPING_AUTHORITY = Tau3GroupingAuthorityContract()

__all__ = [name for name in globals() if name.startswith("TAU3_") or name.startswith("Tau3")]
