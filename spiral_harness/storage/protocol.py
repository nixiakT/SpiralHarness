"""Structural contract for immutable artifact repository backends."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from spiral_harness.core.models import ArtifactRef

T = TypeVar("T")


@runtime_checkable
class ArtifactRepository(Protocol):
    """Minimum trusted storage capability used by orchestration services.

    Implementations may be local or remote, but reads must verify immutable
    references and writes must never replace different content at an existing
    logical identity.
    """

    def put_bytes(
        self,
        data: bytes | bytearray | memoryview,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef: ...

    def put_json(self, value: Any, *, media_type: str = "application/json") -> ArtifactRef: ...

    def get_bytes(self, ref_or_digest: ArtifactRef | str) -> bytes: ...

    def get_json(
        self,
        ref_or_digest: ArtifactRef | str,
        model_type: type[T] | None = None,
    ) -> Any | T: ...


__all__ = ["ArtifactRepository"]
