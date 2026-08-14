"""Publication and verified reload of prospective confirmatory task splits."""

from __future__ import annotations

from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.confirmatory_resources import (
    CONFIRMATORY_TASK_SPLIT_MEDIA_TYPE,
    ConfirmatoryTaskSplitManifest,
)
from spiral_harness.storage.protocol import ArtifactRepository


class ConfirmatoryTaskSplitArtifactError(ValueError):
    """A task-split artifact cannot be joined to its prospective commitment."""


def _require_repository(repository: ArtifactRepository) -> ArtifactRepository:
    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    return repository


def publish_confirmatory_task_split(
    repository: ArtifactRepository,
    manifest: ConfirmatoryTaskSplitManifest,
) -> ArtifactRef:
    """Publish a strictly validated canonical manifest and confirm its exact ref."""

    store = _require_repository(repository)
    checked = ConfirmatoryTaskSplitManifest.model_validate(manifest, strict=True)
    ref = store.put_json(checked, media_type=CONFIRMATORY_TASK_SPLIT_MEDIA_TYPE)
    if ref != checked.artifact_ref:
        raise ConfirmatoryTaskSplitArtifactError(
            "published task-split ref differs from its canonical manifest"
        )
    return ref


def load_confirmatory_task_split(
    repository: ArtifactRepository,
    ref: ArtifactRef,
) -> ConfirmatoryTaskSplitManifest:
    """Reload and validate a content-addressed task-split manifest."""

    store = _require_repository(repository)
    checked_ref = ArtifactRef.model_validate(ref, strict=True)
    if checked_ref.media_type != CONFIRMATORY_TASK_SPLIT_MEDIA_TYPE:
        raise ConfirmatoryTaskSplitArtifactError("task-split ref declares the wrong media type")
    try:
        loaded = store.get_json(checked_ref, ConfirmatoryTaskSplitManifest)
        checked = ConfirmatoryTaskSplitManifest.model_validate(loaded, strict=True)
    except Exception as exc:
        raise ConfirmatoryTaskSplitArtifactError(
            "task-split artifact cannot be loaded and validated"
        ) from exc
    if checked.artifact_ref != checked_ref:
        raise ConfirmatoryTaskSplitArtifactError(
            "loaded task-split content differs from its declared ref"
        )
    return checked


def verify_confirmatory_task_split(
    repository: ArtifactRepository,
    ref: ArtifactRef,
    expected: ConfirmatoryTaskSplitManifest,
) -> ConfirmatoryTaskSplitManifest:
    """Reload a manifest and require equality with an ex-ante expected value."""

    checked_expected = ConfirmatoryTaskSplitManifest.model_validate(expected, strict=True)
    if ref != checked_expected.artifact_ref:
        raise ConfirmatoryTaskSplitArtifactError(
            "task-split ref differs from the expected canonical manifest"
        )
    loaded = load_confirmatory_task_split(repository, ref)
    if loaded != checked_expected:  # pragma: no cover - ref equality implies content equality
        raise ConfirmatoryTaskSplitArtifactError("task-split manifest differs from expected")
    return loaded


__all__ = [
    "ConfirmatoryTaskSplitArtifactError",
    "load_confirmatory_task_split",
    "publish_confirmatory_task_split",
    "verify_confirmatory_task_split",
]
