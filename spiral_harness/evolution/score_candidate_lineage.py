"""Typed proposal-to-harness lineage verification for matched SCORE rounds."""

from __future__ import annotations

from pydantic import BaseModel

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.experiment import CANDIDATE_MANIFEST_MEDIA_TYPE, CandidateManifest
from spiral_harness.core.models import (
    CANDIDATE_MUTATION_MEDIA_TYPE,
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    CandidateMutation,
    HarnessManifest,
    MutationHypothesis,
)
from spiral_harness.evolution.models import PROMPT_PROPOSAL_MEDIA_TYPE, PromptProposal
from spiral_harness.evolution.objective_evidence import TrustedObjectiveAggregateContent
from spiral_harness.evolution.orchestrator import MUTATION_HYPOTHESIS_MEDIA_TYPE
from spiral_harness.experiments.baselines import BaselineKind
from spiral_harness.experiments.matched_v2 import MatchedV2GateQueryBlock, MatchedV2RunManifest
from spiral_harness.storage.protocol import ArtifactRepository


class ScoreCandidateLineageError(ValueError):
    """Raised when a score aggregate cites an incomplete or foreign candidate."""


def _load_exact[ModelT: BaseModel](
    store: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    media_type: str,
) -> ModelT:
    if ref.media_type != media_type:
        raise ScoreCandidateLineageError(f"{model_type.__name__} has wrong media type")
    try:
        payload = store.get_bytes(ref)
        loaded = store.get_json(ref, model_type)
    except Exception as exc:
        raise ScoreCandidateLineageError(
            f"{model_type.__name__} cannot be loaded as canonical content"
        ) from exc
    if payload != canonical_json_bytes(loaded):
        raise ScoreCandidateLineageError(f"{model_type.__name__} is not canonical")
    return loaded


def verify_score_candidate_lineage(
    store: ArtifactRepository,
    *,
    matched_run_ref: ArtifactRef,
    run: MatchedV2RunManifest,
    block: MatchedV2GateQueryBlock,
    objective: TrustedObjectiveAggregateContent,
) -> None:
    """Close exact run, proposal, mutation, and parent/child harness lineage."""

    proposal = _load_exact(
        store,
        objective.proposal_ref,
        PromptProposal,
        PROMPT_PROPOSAL_MEDIA_TYPE,
    )
    candidate = _load_exact(
        store,
        objective.candidate_ref,
        CandidateManifest,
        CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    mutation = _load_exact(
        store,
        candidate.mutation_ref,
        CandidateMutation,
        CANDIDATE_MUTATION_MEDIA_TYPE,
    )
    hypothesis = _load_exact(
        store,
        proposal.hypothesis_ref,
        MutationHypothesis,
        MUTATION_HYPOTHESIS_MEDIA_TYPE,
    )
    parent = _load_exact(
        store,
        candidate.parent_harness_ref,
        HarnessManifest,
        HARNESS_MANIFEST_MEDIA_TYPE,
    )
    child = _load_exact(
        store,
        candidate.child_harness_ref,
        HarnessManifest,
        HARNESS_MANIFEST_MEDIA_TYPE,
    )
    if (
        objective.search_run_ref != matched_run_ref
        or proposal.baseline_kind is not BaselineKind.SCORE_ONLY_MATCHED
        or proposal.round_index != block.query_index
        or proposal.parent_harness_ref != objective.parent_harness_ref
        or candidate.parent_harness_ref != objective.parent_harness_ref
        or candidate.child_harness_ref != objective.candidate_harness_ref
        or proposal.hypothesis_ref not in candidate.evidence_refs
        or mutation.hypothesis != hypothesis
        or proposal.target_component_name != mutation.target_component
        or proposal.before_prompt_ref != mutation.before.artifact
        or proposal.after_prompt_ref != mutation.after.artifact
    ):
        raise ScoreCandidateLineageError("objective candidate lineage is inconsistent")
    parent_components = {component.name: component for component in parent.components}
    child_components = {component.name: component for component in child.components}
    if (
        parent_components.get(mutation.target_component) != mutation.before
        or child_components.get(mutation.target_component) != mutation.after
    ):
        raise ScoreCandidateLineageError("candidate mutation differs from parent/child harnesses")


__all__ = [
    "ScoreCandidateLineageError",
    "verify_score_candidate_lineage",
]
