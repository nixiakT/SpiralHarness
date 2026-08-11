"""Pure strategy validation, finite catalogue sampling, and nomination rules."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import (
    ArtifactRef,
    ComponentKind,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.evolution.models import (
    CANDIDATE_SCREEN_MEDIA_TYPE,
    DIAGNOSIS_MEDIA_TYPE,
    NOMINATION_MEDIA_TYPE,
    PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
    PROMPT_PROPOSAL_MEDIA_TYPE,
    PROPOSAL_BATCH_MEDIA_TYPE,
    CandidateScreen,
    CandidateScreenStatus,
    Nomination,
    PromptProposal,
    SearchPolicy,
    SearchStoppingPolicy,
    StrategyFeedbackView,
    StrategyPluginManifest,
    expected_strategy_feedback,
    expected_strategy_mutation,
)
from spiral_harness.evolution.seeds import (
    RANDOM_VALID_SAMPLE_DOMAIN,
    UNIFORM_SHUFFLE_ALGORITHM,
    derive_random_valid_sample_seed,
    uniform_without_replacement_indices,
)
from spiral_harness.experiments.baselines import (
    BaselineKind,
    FrozenMutationPolicy,
    MutationOperation,
)

RANDOM_VALID_SELECTION_MEDIA_TYPE = "application/vnd.spiral-harness.random-valid-selection.v1+json"
PROMPT_CATALOGUE_SAMPLING_FRAME = "eligible-entries-in-frozen-prompt-catalogue"
PROMPT_CATALOGUE_SAMPLING_CLAIM = "uniform-without-replacement-over-eligible-catalogue-entries"


class StrategyPermissionError(ValueError):
    """Raised when runtime strategy inputs exceed a frozen condition profile."""


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


def _require_text_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if not media_type.startswith("text/"):
        raise ValueError(f"{field_name} must declare a text media type")


def _validated_model[ModelT: ImmutableModel](
    model_type: type[ModelT],
    value: ModelT,
) -> ModelT:
    if not isinstance(value, model_type):
        raise TypeError(f"expected {model_type.__name__}, got {type(value).__name__}")
    return model_type.model_validate(
        value.model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )


def make_search_policy(
    *,
    baseline_kind: BaselineKind,
    mutation_policy: FrozenMutationPolicy,
    max_proposals_per_round: int,
    max_candidates_screened_per_round: int,
    max_rounds: int,
    max_gate_queries: int,
    patience_rounds: int,
    max_consecutive_declines: int,
    max_diagnoses: int,
    max_proposals: int,
    max_screens: int,
    family_alpha: float,
) -> SearchPolicy:
    """Build the only first-version policy profile for a condition."""

    return SearchPolicy(
        baseline_kind=baseline_kind,
        mutation_policy=mutation_policy,
        available_feedback=tuple(expected_strategy_feedback(baseline_kind)),
        mutation=expected_strategy_mutation(baseline_kind),
        max_rounds=max_rounds,
        max_gate_queries=max_gate_queries,
        patience_rounds=patience_rounds,
        max_consecutive_declines=max_consecutive_declines,
        max_diagnoses=max_diagnoses,
        max_proposals=max_proposals,
        max_screens=max_screens,
        max_proposals_per_round=max_proposals_per_round,
        max_candidates_screened_per_round=max_candidates_screened_per_round,
        family_alpha=family_alpha,
        gate_confidence_level=(
            None
            if baseline_kind is BaselineKind.STATIC
            else 1.0 - (family_alpha / max_gate_queries)
        ),
    )


def make_search_stopping_policy(*, policy: SearchPolicy) -> SearchStoppingPolicy:
    """Revalidate a search policy and project its exact stopping artifact."""

    return _validated_model(SearchPolicy, policy).stopping_policy


def make_strategy_plugin_manifest(
    *,
    plugin_id: str,
    plugin_version: str,
    implementation_ref: ArtifactRef,
    baseline_kind: BaselineKind,
) -> StrategyPluginManifest:
    """Build an exact declaration for one of the four strategy profiles."""

    return StrategyPluginManifest(
        plugin_id=plugin_id,
        plugin_version=plugin_version,
        implementation_ref=implementation_ref,
        baseline_kind=baseline_kind,
        consumes_feedback=tuple(expected_strategy_feedback(baseline_kind)),
        mutation=expected_strategy_mutation(baseline_kind),
        uses_finite_prompt_catalogue=baseline_kind is BaselineKind.RANDOM_VALID,
        emits_prompt_proposals=baseline_kind is not BaselineKind.STATIC,
    )


def validate_strategy_permissions(
    *,
    policy: SearchPolicy,
    plugin: StrategyPluginManifest,
    feedback: StrategyFeedbackView | None = None,
) -> tuple[SearchPolicy, StrategyPluginManifest, StrategyFeedbackView | None]:
    """Revalidate and join policy, plugin declarations, and actual disclosure."""

    try:
        checked_policy = _validated_model(SearchPolicy, policy)
        checked_plugin = _validated_model(StrategyPluginManifest, plugin)
        checked_feedback = (
            None if feedback is None else _validated_model(StrategyFeedbackView, feedback)
        )
    except (TypeError, ValueError) as exc:
        raise StrategyPermissionError(f"invalid strategy contract: {exc}") from exc

    if checked_plugin.baseline_kind is not checked_policy.baseline_kind:
        raise StrategyPermissionError("plugin and search policy belong to different baselines")
    if frozenset(checked_plugin.consumes_feedback) != frozenset(checked_policy.available_feedback):
        raise StrategyPermissionError("plugin feedback declaration differs from search policy")
    if checked_plugin.mutation != checked_policy.mutation:
        raise StrategyPermissionError("plugin mutation declaration differs from search policy")
    if checked_feedback is not None:
        if checked_feedback.baseline_kind is not checked_policy.baseline_kind:
            raise StrategyPermissionError(
                "feedback view and search policy belong to different baselines"
            )
        if not frozenset(checked_feedback.exposed_feedback).issubset(
            checked_plugin.consumes_feedback
        ):
            raise StrategyPermissionError("feedback view exceeds plugin permissions")
    return checked_policy, checked_plugin, checked_feedback


class PromptMutationEntry(ImmutableModel):
    """One concrete prompt replacement in a finite frozen catalogue.

    A catalogue may retain an entry that is a no-op for the current prompt so
    its identity does not change as a champion evolves.  Such entries are
    never eligible for sampling.
    """

    schema_version: Literal["1"] = "1"
    entry_id: NonEmptyStr
    target_component_name: NonEmptyStr
    expected_before_prompt_ref: ArtifactRef
    after_prompt_ref: ArtifactRef
    hypothesis_ref: ArtifactRef
    mechanism_family: NonEmptyStr

    @model_validator(mode="after")
    def _validate_entry_refs(self) -> Self:
        _require_text_ref(
            self.expected_before_prompt_ref,
            field_name="expected_before_prompt_ref",
        )
        _require_text_ref(self.after_prompt_ref, field_name="after_prompt_ref")
        _require_json_ref(self.hypothesis_ref, field_name="hypothesis_ref")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def is_noop(self) -> bool:
        return self.expected_before_prompt_ref.sha256 == self.after_prompt_ref.sha256


class PromptMutationCatalogue(ImmutableModel):
    """A finite, enumerable sampling frame for the random-valid baseline."""

    schema_version: Literal["1"] = "1"
    catalogue_id: NonEmptyStr
    grammar_version: NonEmptyStr
    parent_harness_ref: ArtifactRef
    component_kind: Literal[ComponentKind.PROMPT] = ComponentKind.PROMPT
    operation: Literal[MutationOperation.REPLACE] = MutationOperation.REPLACE
    sampling_frame: Literal["eligible-entries-in-frozen-prompt-catalogue"] = (
        PROMPT_CATALOGUE_SAMPLING_FRAME
    )
    entries: Annotated[tuple[PromptMutationEntry, ...], Field(min_length=1)]

    @field_validator("entries")
    @classmethod
    def _canonicalize_entries(
        cls,
        value: tuple[PromptMutationEntry, ...],
    ) -> tuple[PromptMutationEntry, ...]:
        return tuple(sorted(value, key=lambda entry: entry.entry_id))

    @model_validator(mode="after")
    def _validate_catalogue(self) -> Self:
        _require_json_ref(self.parent_harness_ref, field_name="parent_harness_ref")
        entry_ids = tuple(entry.entry_id for entry in self.entries)
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("catalogue entry IDs must be unique")
        coordinates = tuple(
            (
                entry.target_component_name,
                entry.expected_before_prompt_ref.sha256,
                entry.after_prompt_ref.sha256,
            )
            for entry in self.entries
        )
        if len(coordinates) != len(set(coordinates)):
            raise ValueError(
                "catalogue must not duplicate mutations because duplicates bias sampling"
            )
        return self

    @property
    def fingerprint(self) -> str:
        """Canonical identity of the complete finite sampling frame."""

        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        """Return the exact ref produced by canonical storage under the media type."""

        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=self.fingerprint,
            size=len(payload),
            media_type=PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
        )


class RandomValidSelection(ImmutableModel):
    """Auditable output of finite-catalogue sampling.

    The uniform claim is intentionally scoped to the recorded eligible set.
    It makes no distributional claim about an unbounded prompt language.
    """

    schema_version: Literal["1"] = "1"
    catalogue_ref: ArtifactRef
    catalogue_fingerprint: Sha256
    eligible_fingerprint: Sha256
    parent_harness_ref: ArtifactRef
    target_component_name: NonEmptyStr
    current_prompt_ref: ArtifactRef
    strategy_seed: Annotated[int, Field(ge=0, strict=True)]
    round_index: Annotated[int, Field(ge=0, strict=True)]
    sample_seed: Annotated[int, Field(ge=0, strict=True)]
    sample_seed_domain: Literal["spiral-harness/evolution/random-valid-sample/v1"] = (
        RANDOM_VALID_SAMPLE_DOMAIN
    )
    sampling_algorithm: Literal["sha256-counter-fisher-yates-rejection-v1"] = (
        UNIFORM_SHUFFLE_ALGORITHM
    )
    sampling_frame: Literal["eligible-entries-in-frozen-prompt-catalogue"] = (
        PROMPT_CATALOGUE_SAMPLING_FRAME
    )
    sampling_claim: Literal["uniform-without-replacement-over-eligible-catalogue-entries"] = (
        PROMPT_CATALOGUE_SAMPLING_CLAIM
    )
    excluded_entry_ids: tuple[NonEmptyStr, ...] = ()
    eligible_entry_count: Annotated[int, Field(ge=0, strict=True)]
    requested_entry_count: Annotated[int, Field(gt=0, strict=True)]
    selected_entries: tuple[PromptMutationEntry, ...]

    @field_validator("excluded_entry_ids")
    @classmethod
    def _canonicalize_excluded_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("excluded_entry_ids must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        if self.catalogue_ref.sha256 != self.catalogue_fingerprint:
            raise ValueError("catalogue_ref does not match catalogue_fingerprint")
        if self.catalogue_ref.media_type != PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE:
            raise ValueError("catalogue_ref declares the wrong media type")
        _require_json_ref(self.parent_harness_ref, field_name="parent_harness_ref")
        _require_text_ref(self.current_prompt_ref, field_name="current_prompt_ref")
        selected_ids = tuple(entry.entry_id for entry in self.selected_entries)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected entries must be sampled without replacement")
        if set(selected_ids).intersection(self.excluded_entry_ids):
            raise ValueError("selected entries must not contain an excluded catalogue entry")
        if len(self.selected_entries) > self.requested_entry_count:
            raise ValueError("selected entry count exceeds the requested count")
        if len(self.selected_entries) > self.eligible_entry_count:
            raise ValueError("selected entry count exceeds the eligible population")
        expected_selected_count = min(
            self.requested_entry_count,
            self.eligible_entry_count,
        )
        if len(self.selected_entries) != expected_selected_count:
            raise ValueError("selection must contain the full requested eligible sample")
        for entry in self.selected_entries:
            if entry.is_noop:
                raise ValueError("selected random-valid entries must be non-noop")
            if entry.target_component_name != self.target_component_name:
                raise ValueError("selected entry targets a different prompt component")
            if entry.expected_before_prompt_ref != self.current_prompt_ref:
                raise ValueError("selected entry does not apply to the current prompt")
        return self

    @property
    def selected_entry_ids(self) -> tuple[str, ...]:
        return tuple(entry.entry_id for entry in self.selected_entries)


def _eligible_fingerprint(
    *,
    catalogue_fingerprint: str,
    entries: tuple[PromptMutationEntry, ...],
    excluded_entry_ids: tuple[str, ...],
) -> str:
    return canonical_sha256(
        {
            "schema_version": "1",
            "catalogue_fingerprint": catalogue_fingerprint,
            "excluded_entry_ids": excluded_entry_ids,
            "eligible_entry_fingerprints": [entry.fingerprint for entry in entries],
        }
    )


def sample_random_valid(
    *,
    catalogue: PromptMutationCatalogue,
    policy: SearchPolicy,
    seed_harness_ref: ArtifactRef,
    parent_harness_ref: ArtifactRef,
    target_component_name: str,
    current_prompt_ref: ArtifactRef,
    strategy_seed: int,
    round_index: int,
    requested_entry_count: int,
    catalogue_ref: ArtifactRef | None = None,
    excluded_entry_ids: tuple[str, ...] = (),
) -> RandomValidSelection:
    """Sample eligible concrete mutations uniformly without replacement."""

    checked_catalogue = _validated_model(PromptMutationCatalogue, catalogue)
    checked_policy = _validated_model(SearchPolicy, policy)
    if checked_policy.baseline_kind is not BaselineKind.RANDOM_VALID:
        raise StrategyPermissionError("only random-valid may sample the prompt catalogue")
    if checked_policy.mutation_policy.grammar_version != checked_catalogue.grammar_version:
        raise StrategyPermissionError("catalogue grammar_version differs from search policy")
    if checked_policy.mutation_policy.allowed_component_kinds != (ComponentKind.PROMPT,):
        raise StrategyPermissionError("random-valid sampling requires prompt-only grammar")
    if checked_catalogue.parent_harness_ref != seed_harness_ref:
        raise StrategyPermissionError("catalogue belongs to a different seed harness")
    _require_text_ref(current_prompt_ref, field_name="current_prompt_ref")
    if isinstance(requested_entry_count, bool) or not isinstance(requested_entry_count, int):
        raise TypeError("requested_entry_count must be an integer")
    if requested_entry_count <= 0:
        raise ValueError("requested_entry_count must be positive")
    if requested_entry_count > checked_policy.max_proposals_per_round:
        raise StrategyPermissionError("requested sample exceeds max_proposals_per_round")
    canonical_excluded = tuple(sorted(excluded_entry_ids))
    if len(canonical_excluded) != len(set(canonical_excluded)):
        raise StrategyPermissionError("excluded_entry_ids must not contain duplicates")
    catalogue_entry_ids = frozenset(entry.entry_id for entry in checked_catalogue.entries)
    if not set(canonical_excluded).issubset(catalogue_entry_ids):
        raise StrategyPermissionError("excluded_entry_ids contains a foreign catalogue entry")

    eligible = tuple(
        entry
        for entry in checked_catalogue.entries
        if entry.target_component_name == target_component_name
        and entry.expected_before_prompt_ref == current_prompt_ref
        and not entry.is_noop
        and entry.entry_id not in canonical_excluded
        and entry.after_prompt_ref.size <= checked_policy.mutation_policy.max_artifact_size_bytes
    )
    catalogue_fingerprint = checked_catalogue.fingerprint
    eligible_fingerprint = _eligible_fingerprint(
        catalogue_fingerprint=catalogue_fingerprint,
        entries=eligible,
        excluded_entry_ids=canonical_excluded,
    )
    sample_seed = derive_random_valid_sample_seed(
        strategy_seed=strategy_seed,
        round_index=round_index,
        catalogue_fingerprint=catalogue_fingerprint,
        eligible_fingerprint=eligible_fingerprint,
    )
    selected_count = min(requested_entry_count, len(eligible))
    selected_indices = uniform_without_replacement_indices(
        population_size=len(eligible),
        sample_size=selected_count,
        sample_seed=sample_seed,
    )
    selected = tuple(eligible[index] for index in selected_indices)
    expected_catalogue_ref = checked_catalogue.artifact_ref
    bound_catalogue_ref = catalogue_ref or expected_catalogue_ref
    if bound_catalogue_ref != expected_catalogue_ref:
        raise StrategyPermissionError(
            "catalogue_ref is not the canonical frozen catalogue artifact"
        )
    return RandomValidSelection(
        catalogue_ref=bound_catalogue_ref,
        catalogue_fingerprint=catalogue_fingerprint,
        eligible_fingerprint=eligible_fingerprint,
        parent_harness_ref=parent_harness_ref,
        target_component_name=target_component_name,
        current_prompt_ref=current_prompt_ref,
        strategy_seed=strategy_seed,
        round_index=round_index,
        sample_seed=sample_seed,
        excluded_entry_ids=canonical_excluded,
        eligible_entry_count=len(eligible),
        requested_entry_count=requested_entry_count,
        selected_entries=selected,
    )


def proposals_from_random_selection(
    selection: RandomValidSelection,
) -> tuple[PromptProposal, ...]:
    """Materialize the sampled concrete entries into the shared proposal grammar."""

    checked = _validated_model(RandomValidSelection, selection)
    return tuple(
        PromptProposal(
            proposal_id=f"random-valid:{checked.round_index}:{entry.entry_id}",
            baseline_kind=BaselineKind.RANDOM_VALID,
            round_index=checked.round_index,
            parent_harness_ref=checked.parent_harness_ref,
            target_component_name=checked.target_component_name,
            before_prompt_ref=checked.current_prompt_ref,
            after_prompt_ref=entry.after_prompt_ref,
            hypothesis_ref=entry.hypothesis_ref,
            mechanism_family=entry.mechanism_family,
            catalogue_ref=checked.catalogue_ref,
            catalogue_entry_id=entry.entry_id,
        )
        for entry in checked.selected_entries
    )


def nominate_candidate(
    *,
    policy: SearchPolicy,
    screens: tuple[CandidateScreen, ...],
) -> Nomination | None:
    """Nominate at most one eligible candidate by a stable score ordering.

    Proposer confidence is not present in :class:`CandidateScreen` and is
    therefore structurally unavailable to this ranking function.
    """

    checked_policy = _validated_model(SearchPolicy, policy)
    if checked_policy.baseline_kind is BaselineKind.STATIC:
        if screens:
            raise StrategyPermissionError("static cannot have candidate screens")
        return None
    if len(screens) > checked_policy.max_candidates_screened_per_round:
        raise StrategyPermissionError("screen batch exceeds the frozen per-round ceiling")
    checked_screens = tuple(_validated_model(CandidateScreen, screen) for screen in screens)
    if not checked_screens:
        return None

    rounds = {screen.round_index for screen in checked_screens}
    if len(rounds) != 1:
        raise StrategyPermissionError("candidate screens span multiple search rounds")
    if any(screen.baseline_kind is not checked_policy.baseline_kind for screen in checked_screens):
        raise StrategyPermissionError("candidate screen belongs to another baseline")
    proposal_hashes = tuple(screen.proposal_ref.sha256 for screen in checked_screens)
    if len(proposal_hashes) != len(set(proposal_hashes)):
        raise StrategyPermissionError("candidate screens contain duplicate proposal refs")
    candidate_hashes = tuple(
        screen.candidate_ref.sha256
        for screen in checked_screens
        if screen.candidate_ref is not None
    )
    if len(candidate_hashes) != len(set(candidate_hashes)):
        raise StrategyPermissionError("candidate screens contain duplicate candidate refs")

    eligible = tuple(
        screen for screen in checked_screens if screen.status is CandidateScreenStatus.ELIGIBLE
    )
    if not eligible:
        return None
    winner = min(
        eligible,
        key=lambda screen: (
            -screen.confidence_lower,  # type: ignore[operator]
            -screen.mean_delta,  # type: ignore[operator]
            screen.tokens_ratio,
            screen.latency_ratio,
            screen.candidate_ref.sha256,  # type: ignore[union-attr]
            screen.proposal_ref.sha256,
        ),
    )
    # CandidateScreen's eligible invariant proves these values are present.
    assert winner.candidate_ref is not None
    assert winner.candidate_harness_ref is not None
    assert winner.evaluation_ref is not None
    assert winner.primary_score is not None
    assert winner.mean_delta is not None
    assert winner.confidence_lower is not None
    assert winner.regression_rate is not None
    assert winner.tokens_ratio is not None
    assert winner.latency_ratio is not None
    return Nomination(
        baseline_kind=winner.baseline_kind,
        round_index=winner.round_index,
        proposal_ref=winner.proposal_ref,
        candidate_ref=winner.candidate_ref,
        candidate_harness_ref=winner.candidate_harness_ref,
        screen_evaluation_ref=winner.evaluation_ref,
        primary_score=winner.primary_score,
        mean_delta=winner.mean_delta,
        confidence_lower=winner.confidence_lower,
        regression_rate=winner.regression_rate,
        tokens_ratio=winner.tokens_ratio,
        latency_ratio=winner.latency_ratio,
        eligible_candidates_considered=len(eligible),
    )


__all__ = [
    "CANDIDATE_SCREEN_MEDIA_TYPE",
    "DIAGNOSIS_MEDIA_TYPE",
    "NOMINATION_MEDIA_TYPE",
    "PROMPT_CATALOGUE_SAMPLING_CLAIM",
    "PROMPT_CATALOGUE_SAMPLING_FRAME",
    "PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE",
    "PROMPT_PROPOSAL_MEDIA_TYPE",
    "PROPOSAL_BATCH_MEDIA_TYPE",
    "RANDOM_VALID_SELECTION_MEDIA_TYPE",
    "PromptMutationCatalogue",
    "PromptMutationEntry",
    "RandomValidSelection",
    "StrategyPermissionError",
    "make_search_policy",
    "make_search_stopping_policy",
    "make_strategy_plugin_manifest",
    "nominate_candidate",
    "proposals_from_random_selection",
    "sample_random_valid",
    "validate_strategy_permissions",
]
