"""Pure validation and application service for atomic harness mutations."""

from __future__ import annotations

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.experiment import MutationPolicy
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    CandidateMutation,
    HarnessManifest,
    ImmutableModel,
)


class HarnessRegistryError(ValueError):
    """Raised when a candidate cannot be applied under the frozen policy."""


def _validated_model[T: ImmutableModel](value: T) -> T:
    """Revalidate instances so bypassed Pydantic copies fail closed."""

    content = value.model_dump(
        mode="python",
        by_alias=False,
        exclude_none=False,
        round_trip=True,
        warnings="none",
    )
    return type(value).model_validate(content, strict=True, by_name=True)


def _base_media_type(media_type: str) -> str:
    return media_type.partition(";")[0].strip().lower()


class HarnessRegistry:
    """Apply one verified component replacement without storage side effects."""

    def __init__(self, policy: MutationPolicy | None = None) -> None:
        self.policy = _validated_model(policy or MutationPolicy())

    def apply_mutation(
        self,
        *,
        parent: HarnessManifest,
        parent_ref: ArtifactRef,
        mutation: CandidateMutation,
        artifact_bytes: bytes | bytearray | memoryview,
        artifact_media_type: str,
    ) -> HarnessManifest:
        """Validate and apply *mutation*, returning a new manifest.

        The caller supplies bytes rather than a store handle so this service
        cannot perform I/O.  Both the child artifact and the parent manifest
        reference are verified from their actual canonical bytes.
        """

        if not isinstance(artifact_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("artifact_bytes must be bytes-like")
        if not isinstance(artifact_media_type, str) or not artifact_media_type.strip():
            raise TypeError("artifact_media_type must be a non-empty string")

        validated_parent = _validated_model(parent)
        validated_parent_ref = _validated_model(parent_ref)
        validated_mutation = _validated_model(mutation)
        policy = _validated_model(self.policy)
        payload = bytes(artifact_bytes)
        actual_media_type = artifact_media_type.strip()

        self._verify_parent_ref(validated_parent, validated_parent_ref)

        components_by_name = {
            component.name: component for component in validated_parent.components
        }
        actual_before = components_by_name.get(validated_mutation.target_component)
        if actual_before is None:
            raise HarnessRegistryError(
                f"target component {validated_mutation.target_component!r} is absent from parent"
            )
        if actual_before != validated_mutation.before:
            raise HarnessRegistryError(
                "mutation before component does not exactly match the parent manifest"
            )

        self._verify_policy(validated_mutation, policy)
        self._verify_artifact(
            validated_mutation.after.artifact,
            payload,
            actual_media_type,
            policy,
        )

        # Normal validated manifests can share their immutable, unchanged
        # component objects.  A bypassed/non-canonical instance is replaced by
        # its validated form before any sharing occurs.
        component_source = parent if parent == validated_parent else validated_parent
        child_components = tuple(
            validated_mutation.after
            if component.name == validated_mutation.target_component
            else component
            for component in component_source.components
        )
        child = component_source.model_copy(
            update={
                "parent": validated_parent_ref,
                "components": child_components,
            }
        )
        _validated_model(child)
        return child

    @staticmethod
    def _verify_parent_ref(parent: HarnessManifest, parent_ref: ArtifactRef) -> None:
        if parent_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise HarnessRegistryError(
                "parent_ref must declare the exact harness manifest v2 media type"
            )
        parent_bytes = canonical_json_bytes(parent)
        if parent_ref.size != len(parent_bytes):
            raise HarnessRegistryError(
                f"parent_ref size mismatch: expected {len(parent_bytes)}, got {parent_ref.size}"
            )
        actual_digest = sha256_bytes(parent_bytes)
        if parent_ref.sha256 != actual_digest:
            raise HarnessRegistryError(
                f"parent_ref hash mismatch: expected {actual_digest}, got {parent_ref.sha256}"
            )

    @staticmethod
    def _verify_policy(mutation: CandidateMutation, policy: MutationPolicy) -> None:
        component = mutation.after
        if component.kind not in policy.allowed_kinds:
            raise HarnessRegistryError(
                f"component kind {component.kind.value!r} is not allowed by mutation policy"
            )
        if (
            policy.allowed_component_names is not None
            and component.name not in policy.allowed_component_names
        ):
            raise HarnessRegistryError(
                f"component name {component.name!r} is not allowed by mutation policy"
            )

    @staticmethod
    def _verify_artifact(
        expected: ArtifactRef,
        payload: bytes,
        actual_media_type: str,
        policy: MutationPolicy,
    ) -> None:
        if actual_media_type != expected.media_type:
            raise HarnessRegistryError(
                "artifact media type does not match the candidate after reference"
            )
        if (
            policy.allowed_media_types is not None
            and _base_media_type(actual_media_type) not in policy.allowed_media_types
        ):
            raise HarnessRegistryError(
                f"artifact media type {actual_media_type!r} is not allowed by mutation policy"
            )
        if len(payload) != expected.size:
            raise HarnessRegistryError(
                f"artifact size mismatch: expected {expected.size}, got {len(payload)}"
            )
        if (
            policy.max_artifact_size_bytes is not None
            and len(payload) > policy.max_artifact_size_bytes
        ):
            raise HarnessRegistryError(
                "artifact exceeds the mutation policy size ceiling: "
                f"{len(payload)} > {policy.max_artifact_size_bytes}"
            )
        actual_digest = sha256_bytes(payload)
        if actual_digest != expected.sha256:
            raise HarnessRegistryError(
                f"artifact hash mismatch: expected {expected.sha256}, got {actual_digest}"
            )


__all__ = ["HarnessRegistry", "HarnessRegistryError"]
