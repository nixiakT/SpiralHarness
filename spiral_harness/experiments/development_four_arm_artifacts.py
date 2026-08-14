"""Typed harness publication for the non-reportable four-arm development run."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    BudgetPolicy,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    ImmutableModel,
    Sha256,
)
from spiral_harness.execution.contracts import FrozenModelSpec, ResolvedHarness
from spiral_harness.experiments.development_four_arm_contracts import DevelopmentArm
from spiral_harness.storage.protocol import ArtifactRepository

DEVELOPMENT_TREATMENT_BINDING_MEDIA_TYPE = (
    "application/vnd.spiral-harness.development-four-arm-treatment.v1+json"
)
DEVELOPMENT_TRUSTED_PLANE_VERSION = "spiral-harness/development-four-arm-runner/v1"


class DevelopmentTreatmentBinding(ImmutableModel):
    """Condition metadata bound into a custom-executor harness manifest."""

    schema_version: Literal["1"] = "1"
    adaptive_stage_fingerprint: Sha256
    condition_id: Sha256
    arm: DevelopmentArm
    purpose: Literal["solver", "proposer"]
    feedback_view: Literal["none", "aggregate-score", "fit-item-evidence"]
    promotion_rule: Literal["none", "development-automatic-fit-v1"]

    @model_validator(mode="after")
    def exact_treatment_matrix(self) -> Self:
        coordinate = (
            self.arm,
            self.purpose,
            self.feedback_view,
            self.promotion_rule,
        )
        allowed = {
            (DevelopmentArm.STATIC, "solver", "none", "none"),
            (
                DevelopmentArm.SCORE,
                "solver",
                "aggregate-score",
                "development-automatic-fit-v1",
            ),
            (
                DevelopmentArm.FULL,
                "solver",
                "fit-item-evidence",
                "development-automatic-fit-v1",
            ),
            (
                DevelopmentArm.SCORE,
                "proposer",
                "aggregate-score",
                "development-automatic-fit-v1",
            ),
            (
                DevelopmentArm.FULL,
                "proposer",
                "fit-item-evidence",
                "development-automatic-fit-v1",
            ),
        }
        if coordinate not in allowed:
            raise ValueError("development treatment coordinate is not in the frozen matrix")
        return self


def publish_development_harness(
    repository: ArtifactRepository,
    *,
    spec: FrozenModelSpec,
    adaptive_stage_fingerprint: str,
    condition_id: str,
    arm: DevelopmentArm,
    purpose: Literal["solver", "proposer"],
    feedback_view: Literal["none", "aggregate-score", "fit-item-evidence"],
    promotion_rule: Literal["none", "development-automatic-fit-v1"],
    system_prompt: str,
) -> ResolvedHarness:
    """Publish and reload one valid manifest before materializing its exact prompt."""

    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    binding = DevelopmentTreatmentBinding(
        adaptive_stage_fingerprint=adaptive_stage_fingerprint,
        condition_id=condition_id,
        arm=arm,
        purpose=purpose,
        feedback_view=feedback_view,
        promotion_rule=promotion_rule,
    )
    prompt_ref = repository.put_bytes(
        system_prompt.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
    )
    binding_ref = repository.put_json(
        binding,
        media_type=DEVELOPMENT_TREATMENT_BINDING_MEDIA_TYPE,
    )
    manifest = HarnessManifest(
        model_fingerprint=checked_spec.model_fingerprint,
        runtime_fingerprint=checked_spec.runtime_fingerprint,
        trusted_plane_version=DEVELOPMENT_TRUSTED_PLANE_VERSION,
        components=(
            HarnessComponentRef(
                name="system_prompt",
                kind=ComponentKind.PROMPT,
                artifact=prompt_ref,
            ),
            HarnessComponentRef(
                name="development_treatment",
                kind=ComponentKind.CONTROL_FLOW,
                artifact=binding_ref,
            ),
        ),
        budget=BudgetPolicy(max_tokens=checked_spec.inference.max_output_tokens),
    )
    raw_ref = repository.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    manifest_ref = ArtifactRef.model_validate(raw_ref, strict=True)
    if repository.get_json(manifest_ref, HarnessManifest) != manifest:
        raise RuntimeError("persisted development harness manifest changed")
    return ResolvedHarness.from_prompt(
        harness_ref=manifest_ref,
        system_prompt=system_prompt,
    )


__all__ = [
    "DEVELOPMENT_TREATMENT_BINDING_MEDIA_TYPE",
    "DEVELOPMENT_TRUSTED_PLANE_VERSION",
    "DevelopmentTreatmentBinding",
    "publish_development_harness",
]
