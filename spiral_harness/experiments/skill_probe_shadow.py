"""Bind a non-promoting shadow report to an already verified closure."""

from __future__ import annotations

from pydantic import BaseModel

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.storage.protocol import ArtifactRepository

from .skill_probe_closure import (
    SKILL_PROBE_SHADOW_REPORT_MEDIA_TYPE,
    MatchedSkillProbeClosure,
    SkillProbeShadowReport,
)


class SkillProbeShadowVerificationError(RuntimeError):
    """Raised when a shadow report is not canonical or closure-bound."""


def _load_exact[ModelT: BaseModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    *,
    media_type: str,
    label: str,
) -> ModelT:
    try:
        checked_ref = ArtifactRef.model_validate(ref, strict=True)
        if checked_ref.media_type != media_type:
            raise ValueError("wrong media type")
        payload = repository.get_bytes(checked_ref)
        loaded = repository.get_json(checked_ref, model_type)
        checked = model_type.model_validate(loaded, strict=True)
        canonical = canonical_json_bytes(checked)
    except Exception as exc:
        raise SkillProbeShadowVerificationError(f"{label} cannot be loaded exactly") from exc
    if (
        payload != canonical
        or len(payload) != checked_ref.size
        or sha256_bytes(payload) != checked_ref.sha256
    ):
        raise SkillProbeShadowVerificationError(f"{label} is not canonical under its reference")
    return checked


def _verify_shadow_for_closure(
    repository: ArtifactRepository,
    *,
    shadow_report_ref: ArtifactRef,
    verified_closure_ref: ArtifactRef,
    verified_closure: MatchedSkillProbeClosure,
) -> SkillProbeShadowReport:
    """Structurally bind a shadow to a closure already verified with live ledgers."""

    report = _load_exact(
        repository,
        shadow_report_ref,
        SkillProbeShadowReport,
        media_type=SKILL_PROBE_SHADOW_REPORT_MEDIA_TYPE,
        label="skill probe shadow report",
    )
    if report.execution_closure_ref != verified_closure_ref:
        raise SkillProbeShadowVerificationError("shadow report belongs to another closure")
    expected = (
        verified_closure.authorization_ref,
        verified_closure.plan_ref,
        verified_closure.running_probes_tail_ref,
        (
            verified_closure.revert.request_inclusion_ref,
            verified_closure.placebo.request_inclusion_ref,
        ),
    )
    actual = (
        report.authorization_ref,
        report.plan_ref,
        report.running_probes_tail_ref,
        report.request_inclusion_refs,
    )
    if actual != expected:
        raise SkillProbeShadowVerificationError("shadow report context differs from its closure")
    return report


__all__ = ["SkillProbeShadowVerificationError"]
