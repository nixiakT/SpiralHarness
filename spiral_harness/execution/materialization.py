"""Trusted resolution of immutable harness manifests into model inputs.

The fixed runner accepts only a :class:`ResolvedHarness`.  This service is the
repository boundary that constructs one from an exact content-addressed
manifest.  It admits exactly one prompt and at most one declarative skill;
unsupported, missing, or additional components fail before an attempt can be
reserved.
"""

from __future__ import annotations

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessManifest,
)
from spiral_harness.execution.contracts import (
    FrozenModelSpec,
    ModelExecution,
    ResolvedHarness,
)
from spiral_harness.skills.loading import (
    SkillDisclosureLevel,
    SkillPackageError,
    SkillPackageLoader,
)
from spiral_harness.skills.package import RESERVED_SKILL_CONTEXT_DELIMITERS
from spiral_harness.storage.protocol import ArtifactRepository


class HarnessMaterializationError(ValueError):
    """Raised when an immutable harness cannot become a trusted model input."""


class HarnessMaterializer:
    """Strictly load one prompt and an optional single compatible skill."""

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        spec: FrozenModelSpec,
    ) -> None:
        if not isinstance(repository, ArtifactRepository):
            raise TypeError("repository must implement ArtifactRepository")
        self.__repository = repository
        self._spec = FrozenModelSpec.model_validate(spec, strict=True)
        self._skill_loader = SkillPackageLoader(repository)

    @property
    def spec(self) -> FrozenModelSpec:
        """Return the immutable model/runtime compatibility target."""

        return self._spec

    def materialize(self, harness_ref: ArtifactRef) -> ResolvedHarness:
        """Resolve one exact manifest without performing an execution attempt."""

        checked_ref = self._validate_harness_ref(harness_ref)
        manifest = self._load_harness(checked_ref)
        self._verify_execution_compatibility(manifest)

        prompts = tuple(
            component for component in manifest.components if component.kind is ComponentKind.PROMPT
        )
        skills = tuple(
            component for component in manifest.components if component.kind is ComponentKind.SKILL
        )
        unsupported = tuple(
            component
            for component in manifest.components
            if component.kind not in {ComponentKind.PROMPT, ComponentKind.SKILL}
        )
        if unsupported:
            kinds = ", ".join(sorted({component.kind.value for component in unsupported}))
            raise HarnessMaterializationError(
                f"harness contains unsupported execution component kind(s): {kinds}"
            )
        if len(prompts) != 1:
            raise HarnessMaterializationError(
                f"harness must contain exactly one prompt component; found {len(prompts)}"
            )
        if len(skills) > 1:
            raise HarnessMaterializationError(
                f"harness may contain at most one skill component; found {len(skills)}"
            )

        base_system_prompt = self._load_prompt(prompts[0].artifact)
        if not skills:
            return ResolvedHarness.from_prompt(
                harness_ref=checked_ref,
                system_prompt=base_system_prompt,
            )

        skill_component = skills[0]
        try:
            disclosure = self._skill_loader.disclose(
                skill_component.artifact,
                level=SkillDisclosureLevel.RULES,
                model_fingerprint=self._spec.model_fingerprint,
                runtime_fingerprint=self._spec.runtime_fingerprint,
            )
        except SkillPackageError as exc:
            raise HarnessMaterializationError(
                f"skill component could not be materialized: {exc}"
            ) from exc
        if disclosure.skill_id != skill_component.name:
            raise HarnessMaterializationError(
                "skill package identity does not match its harness component name"
            )
        return ResolvedHarness.from_skill(
            harness_ref=checked_ref,
            base_system_prompt=base_system_prompt,
            skill_disclosure=disclosure,
        )

    def verify_execution_request(
        self,
        harness_ref: ArtifactRef,
        execution: ModelExecution,
    ) -> ResolvedHarness:
        """Replay one exact harness and rejoin it to the actual model request.

        This method establishes exact request inclusion only.  It accepts no
        caller-authored mechanism booleans: it reloads the manifest and skill
        package, recreates the disclosure, and requires every prompt field in
        the persisted execution to equal that deterministic result.  It does
        not prove provider delivery, model activation, adherence, or benefit.
        """

        expected = self.materialize(harness_ref)
        try:
            checked_execution = ModelExecution.model_validate(
                execution.model_dump(
                    mode="python",
                    round_trip=True,
                    warnings="none",
                ),
                strict=True,
            )
        except Exception as exc:
            raise HarnessMaterializationError(
                f"model execution could not be verified: {exc}"
            ) from exc
        if checked_execution.spec != self._spec:
            raise HarnessMaterializationError(
                "model execution context differs from the materializer's frozen spec"
            )

        request = checked_execution.request
        actual = ResolvedHarness(
            harness_ref=request.harness_ref,
            base_system_prompt=request.base_system_prompt,
            base_system_prompt_sha256=request.base_system_prompt_sha256,
            skill_disclosure=request.skill_disclosure,
            system_prompt=request.system_prompt,
            resolved_prompt_sha256=request.resolved_prompt_sha256,
        )
        if actual != expected:
            raise HarnessMaterializationError(
                "model execution request differs from the exact materialized harness"
            )
        return expected

    @staticmethod
    def _validate_harness_ref(harness_ref: ArtifactRef) -> ArtifactRef:
        try:
            checked = ArtifactRef.model_validate(harness_ref, strict=True)
        except Exception as exc:
            raise HarnessMaterializationError("harness reference is malformed") from exc
        if checked.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise HarnessMaterializationError(
                "harness reference must declare the exact harness manifest v2 media type"
            )
        return checked

    def _load_harness(self, harness_ref: ArtifactRef) -> HarnessManifest:
        try:
            payload = self.__repository.get_bytes(harness_ref)
            loaded = self.__repository.get_json(harness_ref, HarnessManifest)
            manifest = HarnessManifest.model_validate(
                loaded.model_dump(
                    mode="python",
                    by_alias=False,
                    exclude_none=False,
                    round_trip=True,
                    warnings="none",
                ),
                strict=True,
            )
        except Exception as exc:
            raise HarnessMaterializationError(
                f"harness manifest could not be verified: {exc}"
            ) from exc
        canonical = canonical_json_bytes(manifest)
        if payload != canonical:
            raise HarnessMaterializationError("harness manifest is not canonical JSON")
        if harness_ref.size != len(canonical):
            raise HarnessMaterializationError("harness reference size does not match content")
        if harness_ref.sha256 != sha256_bytes(canonical):
            raise HarnessMaterializationError("harness reference hash does not match content")
        return manifest

    def _verify_execution_compatibility(self, manifest: HarnessManifest) -> None:
        if manifest.model_fingerprint != self._spec.model_fingerprint:
            raise HarnessMaterializationError(
                "harness model fingerprint differs from the frozen model"
            )
        if manifest.runtime_fingerprint != self._spec.runtime_fingerprint:
            raise HarnessMaterializationError(
                "harness runtime fingerprint differs from the frozen runtime"
            )

    def _load_prompt(self, prompt_ref: ArtifactRef) -> str:
        media_type = prompt_ref.media_type.partition(";")[0].strip().lower()
        if media_type != "text/plain":
            raise HarnessMaterializationError(
                "prompt component must declare the text/plain media type"
            )
        try:
            payload = self.__repository.get_bytes(prompt_ref)
        except Exception as exc:
            raise HarnessMaterializationError(
                f"prompt component could not be verified: {exc}"
            ) from exc
        if prompt_ref.size != len(payload):
            raise HarnessMaterializationError("prompt reference size does not match content")
        if prompt_ref.sha256 != sha256_bytes(payload):
            raise HarnessMaterializationError("prompt reference hash does not match content")
        try:
            prompt = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise HarnessMaterializationError(
                "prompt component must contain exact UTF-8 text"
            ) from exc
        if not prompt:
            raise HarnessMaterializationError("prompt component must not be empty")
        if any(delimiter in prompt for delimiter in RESERVED_SKILL_CONTEXT_DELIMITERS):
            raise HarnessMaterializationError(
                "prompt component contains a reserved skill context delimiter"
            )
        return prompt


__all__ = ["HarnessMaterializationError", "HarnessMaterializer"]
