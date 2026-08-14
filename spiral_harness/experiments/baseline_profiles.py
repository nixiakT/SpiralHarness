"""Fail-closed condition profiles for the five-arm score-versus-evidence study.

Protocol v2 is a design-time boundary, not runtime admission.  SCORE and FULL
share one action grammar and capability; only their typed feedback grants may
differ.  The matched-contrast report is structural and explicitly does not
attest execution topology or runtime seed use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Self

from pydantic import field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ComponentKind, ImmutableModel, Sha256
from spiral_harness.experiments.baselines import (
    BaselineKind,
    FeedbackType,
    FrozenMutationPolicy,
)

BASELINE_PROFILE_VERSION = "score-only-matched-v2"
PAIRED_PROPOSER_GROUP = "score-full"
V2_BASELINE_KINDS = (
    BaselineKind.STATIC,
    BaselineKind.RANDOM_VALID,
    BaselineKind.PROMPT_ONLY,
    BaselineKind.SCORE_ONLY_MATCHED,
    BaselineKind.EVIDENCE_TARGETED,
)
SCORE_FULL_KINDS = (
    BaselineKind.SCORE_ONLY_MATCHED,
    BaselineKind.EVIDENCE_TARGETED,
)


class BaselineProfileError(ValueError):
    """Raised when a condition has no exact protocol-v2 profile."""


class ActionSelectionMode(StrEnum):
    """How a v2 condition chooses actions from the common frozen grammar."""

    NONE = "none"
    UNIFORM_RANDOM_VALID = "uniform-random-valid"
    PROMPT_OPTIMIZATION = "prompt-optimization"
    MATCHED_OPTIMIZATION = "matched-optimization"


class V2ActionCapability(ImmutableModel):
    """Evidence-agnostic action authority for protocol v2.

    Evidence visibility is intentionally absent.  It is represented only by
    ``available_feedback`` on :class:`BaselineConditionProfile`.
    """

    selection_mode: ActionSelectionMode
    mutable_component_kinds: tuple[ComponentKind, ...] = ()
    may_call_optimizer_model: bool = False

    @field_validator("mutable_component_kinds")
    @classmethod
    def _canonicalize_component_kinds(
        cls,
        value: tuple[ComponentKind, ...],
    ) -> tuple[ComponentKind, ...]:
        ordered = tuple(sorted(value, key=lambda kind: kind.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("mutable_component_kinds must not contain duplicates")
        return ordered


_COMMON_SEARCH_FEEDBACK = (
    FeedbackType.BENCHMARK_METADATA,
    FeedbackType.EXPLORATION_INPUTS,
    FeedbackType.GATE_AGGREGATES,
)
_SCORE_AGGREGATE_FEEDBACK = (
    *_COMMON_SEARCH_FEEDBACK,
    FeedbackType.EXPLORATION_AGGREGATES,
)
_FEEDBACK_PROFILES: Mapping[BaselineKind, tuple[FeedbackType, ...]] = MappingProxyType(
    {
        BaselineKind.STATIC: (FeedbackType.BENCHMARK_METADATA,),
        BaselineKind.RANDOM_VALID: _COMMON_SEARCH_FEEDBACK,
        BaselineKind.PROMPT_ONLY: (
            *_SCORE_AGGREGATE_FEEDBACK,
            FeedbackType.EXPLORATION_ITEM_FEEDBACK,
            FeedbackType.EXPLORATION_TRAJECTORIES,
        ),
        BaselineKind.SCORE_ONLY_MATCHED: _SCORE_AGGREGATE_FEEDBACK,
        BaselineKind.EVIDENCE_TARGETED: (
            *_SCORE_AGGREGATE_FEEDBACK,
            FeedbackType.EXPLORATION_ITEM_FEEDBACK,
            FeedbackType.EXPLORATION_TRAJECTORIES,
            FeedbackType.DIAGNOSTIC_EVIDENCE,
            FeedbackType.MECHANISM_EVIDENCE,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class _ActionSpec:
    selection_mode: ActionSelectionMode
    component_scope: Literal["none", "prompt", "full"]
    may_call_optimizer_model: bool = False


_MATCHED_OPTIMIZER_SPEC = _ActionSpec(
    selection_mode=ActionSelectionMode.MATCHED_OPTIMIZATION,
    component_scope="full",
    may_call_optimizer_model=True,
)
_ACTION_PROFILES: Mapping[BaselineKind, _ActionSpec] = MappingProxyType(
    {
        BaselineKind.STATIC: _ActionSpec(ActionSelectionMode.NONE, "none"),
        BaselineKind.RANDOM_VALID: _ActionSpec(
            ActionSelectionMode.UNIFORM_RANDOM_VALID,
            "full",
        ),
        BaselineKind.PROMPT_ONLY: _ActionSpec(
            ActionSelectionMode.PROMPT_OPTIMIZATION,
            "prompt",
            may_call_optimizer_model=True,
        ),
        BaselineKind.SCORE_ONLY_MATCHED: _MATCHED_OPTIMIZER_SPEC,
        BaselineKind.EVIDENCE_TARGETED: _MATCHED_OPTIMIZER_SPEC,
    }
)


def _require_complete_mapping[ValueT](
    mapping_name: str,
    mapping: Mapping[BaselineKind, ValueT],
) -> None:
    supplied = frozenset(mapping)
    expected = frozenset(BaselineKind)
    if supplied != expected or len(mapping) != len(BaselineKind):
        missing = sorted(kind.value for kind in expected.difference(supplied))
        unexpected = sorted(kind.value for kind in supplied.difference(expected))
        raise RuntimeError(
            f"{mapping_name} must explicitly cover every BaselineKind; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


if frozenset(V2_BASELINE_KINDS) != frozenset(BaselineKind) or len(V2_BASELINE_KINDS) != len(
    BaselineKind
):
    raise RuntimeError("V2_BASELINE_KINDS must explicitly order every BaselineKind")
_require_complete_mapping("feedback profiles", _FEEDBACK_PROFILES)
_require_complete_mapping("action profiles", _ACTION_PROFILES)


def _require_known_kind(kind: BaselineKind | object) -> BaselineKind:
    if type(kind) is not BaselineKind:
        raise TypeError("kind must be an exact BaselineKind")
    if kind not in _FEEDBACK_PROFILES or kind not in _ACTION_PROFILES:
        raise BaselineProfileError(f"unsupported baseline condition: {kind.value}")
    return kind


def feedback_profile(kind: BaselineKind) -> tuple[FeedbackType, ...]:
    """Return one exact, canonical feedback grant; unknown conditions fail closed."""

    checked = _require_known_kind(kind)
    try:
        values = _FEEDBACK_PROFILES[checked]
    except KeyError as exc:
        raise BaselineProfileError(f"no feedback profile for {checked.value}") from exc
    return tuple(sorted(values, key=lambda value: value.value))


def action_capability_profile(
    kind: BaselineKind,
    mutation_policy: FrozenMutationPolicy,
) -> V2ActionCapability:
    """Project evidence-agnostic action authority from the common grammar."""

    checked = _require_known_kind(kind)
    if type(mutation_policy) is not FrozenMutationPolicy:
        raise TypeError("mutation_policy must be an exact FrozenMutationPolicy")
    policy = FrozenMutationPolicy.model_validate(
        mutation_policy.model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )
    try:
        spec = _ACTION_PROFILES[checked]
    except KeyError as exc:
        raise BaselineProfileError(f"no action profile for {checked.value}") from exc
    if spec.component_scope == "none":
        mutable_kinds: tuple[ComponentKind, ...] = ()
    elif spec.component_scope == "prompt":
        if ComponentKind.PROMPT not in policy.allowed_component_kinds:
            raise BaselineProfileError("prompt-only requires prompt in the frozen grammar")
        mutable_kinds = (ComponentKind.PROMPT,)
    elif spec.component_scope == "full":
        mutable_kinds = policy.allowed_component_kinds
    else:  # pragma: no cover - closed Literal plus import-time mapping checks.
        raise BaselineProfileError(f"unknown component scope: {spec.component_scope}")
    return V2ActionCapability(
        selection_mode=spec.selection_mode,
        mutable_component_kinds=mutable_kinds,
        may_call_optimizer_model=spec.may_call_optimizer_model,
    )


class BaselineConditionProfile(ImmutableModel):
    """Persistable five-arm condition profile for future v2 admission."""

    schema_version: Literal["2"] = "2"
    profile_version: Literal["score-only-matched-v2"] = BASELINE_PROFILE_VERSION
    kind: BaselineKind
    mutation_policy: FrozenMutationPolicy
    available_feedback: tuple[FeedbackType, ...]
    action_capability: V2ActionCapability
    paired_proposer_group: Literal["score-full"] | None = None

    @field_validator("available_feedback")
    @classmethod
    def _canonicalize_feedback(
        cls,
        value: tuple[FeedbackType, ...],
    ) -> tuple[FeedbackType, ...]:
        ordered = tuple(sorted(value, key=lambda feedback: feedback.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("available_feedback must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _enforce_exact_profile(self) -> Self:
        if self.available_feedback != feedback_profile(self.kind):
            raise ValueError(f"{self.kind.value} feedback differs from its v2 profile")
        if self.action_capability != action_capability_profile(self.kind, self.mutation_policy):
            raise ValueError(f"{self.kind.value} action capability differs from its v2 profile")
        expected_group = PAIRED_PROPOSER_GROUP if self.kind in SCORE_FULL_KINDS else None
        if self.paired_proposer_group != expected_group:
            raise ValueError(f"{self.kind.value} has the wrong proposer pairing group")
        return self


def make_condition_profile(
    *,
    kind: BaselineKind,
    mutation_policy: FrozenMutationPolicy,
) -> BaselineConditionProfile:
    """Build the sole protocol-v2 profile for one explicitly mapped condition."""

    checked = _require_known_kind(kind)
    return BaselineConditionProfile(
        kind=checked,
        mutation_policy=mutation_policy,
        available_feedback=feedback_profile(checked),
        action_capability=action_capability_profile(checked, mutation_policy),
        paired_proposer_group=(PAIRED_PROPOSER_GROUP if checked in SCORE_FULL_KINDS else None),
    )


class MatchedContrastProfile(ImmutableModel):
    """Pair-level profile allowing only kind and feedback-grant differences.

    This guarantee is confined to fields in the structural profile schema; it
    does not cover runtime model calls, budgets, topology, or artifact content.
    """

    schema_version: Literal["1"] = "1"
    score: BaselineConditionProfile
    full: BaselineConditionProfile
    allowed_treatment_difference: Literal["kind-and-feedback-grant-only"] = (
        "kind-and-feedback-grant-only"
    )

    @model_validator(mode="after")
    def _enforce_matched_contrast(self) -> Self:
        if self.score.kind is not BaselineKind.SCORE_ONLY_MATCHED:
            raise ValueError("score profile must use score-only-matched")
        if self.full.kind is not BaselineKind.EVIDENCE_TARGETED:
            raise ValueError("full profile must use evidence-targeted")
        if self.score.mutation_policy != self.full.mutation_policy:
            raise ValueError("SCORE and FULL mutation policies must be identical")
        if self.score.action_capability != self.full.action_capability:
            raise ValueError("SCORE and FULL action capabilities must be identical")
        score_non_treatment = self.score.model_dump(
            mode="python",
            exclude={"kind", "available_feedback"},
            round_trip=True,
            warnings="none",
        )
        full_non_treatment = self.full.model_dump(
            mode="python",
            exclude={"kind", "available_feedback"},
            round_trip=True,
            warnings="none",
        )
        if score_non_treatment != full_non_treatment:
            raise ValueError(
                "within the profile schema SCORE and FULL may differ only in kind and feedback"
            )
        if self.score.available_feedback == self.full.available_feedback:
            raise ValueError("SCORE and FULL must preserve the frozen feedback treatment")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def make_matched_contrast_profile(
    *,
    mutation_policy: FrozenMutationPolicy,
) -> MatchedContrastProfile:
    """Build the structurally matched SCORE/FULL pair."""

    return MatchedContrastProfile(
        score=make_condition_profile(
            kind=BaselineKind.SCORE_ONLY_MATCHED,
            mutation_policy=mutation_policy,
        ),
        full=make_condition_profile(
            kind=BaselineKind.EVIDENCE_TARGETED,
            mutation_policy=mutation_policy,
        ),
    )


_MATCHED_CONTRAST_CHECKS = (
    "mutation-policy-equal",
    "action-grammar-equal",
    "action-capability-equal",
    "only-kind-and-feedback-grant-differ",
)


class MatchedContrastReport(ImmutableModel):
    """Structural phase1 report; it is not execution or runtime-seed evidence."""

    schema_version: Literal["1"] = "1"
    contrast: MatchedContrastProfile
    contrast_fingerprint: Sha256
    checks: tuple[
        Literal["mutation-policy-equal"],
        Literal["action-grammar-equal"],
        Literal["action-capability-equal"],
        Literal["only-kind-and-feedback-grant-differ"],
    ] = _MATCHED_CONTRAST_CHECKS
    structurally_matched: Literal[True] = True
    execution_attested: Literal[False] = False
    runtime_topology_matched: Literal[False] = False
    paired_proposer_seed_runtime_bound: Literal[False] = False

    @model_validator(mode="after")
    def _bind_exact_structural_report(self) -> Self:
        if self.contrast_fingerprint != self.contrast.fingerprint:
            raise ValueError("contrast_fingerprint does not match the contrast profile")
        if self.checks != _MATCHED_CONTRAST_CHECKS:
            raise ValueError("matched contrast checks must be the complete canonical set")
        return self


def make_matched_contrast_report(
    *,
    mutation_policy: FrozenMutationPolicy,
) -> MatchedContrastReport:
    """Validate the pair and report only phase1 structural guarantees."""

    contrast = make_matched_contrast_profile(mutation_policy=mutation_policy)
    return MatchedContrastReport(
        contrast=contrast,
        contrast_fingerprint=contrast.fingerprint,
    )


__all__ = [
    "BASELINE_PROFILE_VERSION",
    "PAIRED_PROPOSER_GROUP",
    "SCORE_FULL_KINDS",
    "V2_BASELINE_KINDS",
    "ActionSelectionMode",
    "BaselineConditionProfile",
    "BaselineProfileError",
    "MatchedContrastProfile",
    "MatchedContrastReport",
    "V2ActionCapability",
    "action_capability_profile",
    "feedback_profile",
    "make_condition_profile",
    "make_matched_contrast_profile",
    "make_matched_contrast_report",
]
