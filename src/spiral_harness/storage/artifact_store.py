"""A small immutable, content-addressed artifact store."""

from __future__ import annotations

import errno
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = {
    errno.EINVAL,
    getattr(errno, "ENOSYS", errno.EINVAL),
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}
T = TypeVar("T")


class ArtifactStoreError(RuntimeError):
    """Base error for artifact persistence and retrieval."""


class ArtifactNotFoundError(ArtifactStoreError, FileNotFoundError):
    """Raised when a referenced artifact is absent."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when stored bytes do not match their immutable reference."""


class ArtifactDecodeError(ArtifactStoreError, ValueError):
    """Raised when referenced bytes are not valid canonical UTF-8 JSON."""


class ArtifactMediaTypeError(ArtifactStoreError, ValueError):
    """Raised when a typed reference does not declare JSON content."""


class ArtifactStore:
    """Persist artifacts under their SHA-256 digest.

    Objects are sharded by the first two digest characters.  Writes use a
    temporary file and an atomic hard link, so an existing object is never
    overwritten.  Every read recomputes the digest; filesystem modification is
    therefore detected before bytes reach an evaluator or proposer.

    A successful POSIX write also fsyncs the shard directory.  Python does not
    expose a portable directory flush on every platform (notably Windows); on
    those platforms the file data is flushed and publication remains atomic,
    while power-loss persistence of the directory entry follows the filesystem's
    guarantees.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.objects_dir = self.root / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self._fsync_directory(self.root)

    @staticmethod
    def _fsync_directory(directory: Path) -> bool:
        """Flush a directory when supported, returning whether it was flushed."""

        if os.name == "nt":
            return False

        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(directory, flags)
        except OSError as exc:
            if exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                return False
            raise ArtifactStoreError(
                f"cannot open artifact directory for fsync: {directory}"
            ) from exc

        try:
            try:
                os.fsync(descriptor)
            except OSError as exc:
                if exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                    return False
                raise ArtifactStoreError(f"cannot fsync artifact directory: {directory}") from exc
        finally:
            os.close(descriptor)
        return True

    @staticmethod
    def _digest(ref_or_digest: ArtifactRef | str) -> str:
        digest = ref_or_digest.sha256 if isinstance(ref_or_digest, ArtifactRef) else ref_or_digest
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("artifact digest must be 64 lowercase hexadecimal characters")
        return digest

    def path_for(self, ref_or_digest: ArtifactRef | str) -> Path:
        """Return the object path for diagnostics and store maintenance."""

        digest = self._digest(ref_or_digest)
        return self.objects_dir / digest[:2] / digest[2:]

    @staticmethod
    def _verify(data: bytes, ref: ArtifactRef) -> None:
        if len(data) != ref.size:
            raise ArtifactIntegrityError(
                f"artifact {ref.sha256} size mismatch: expected {ref.size}, got {len(data)}"
            )
        actual = sha256_bytes(data)
        if actual != ref.sha256:
            raise ArtifactIntegrityError(
                f"artifact {ref.sha256} hash mismatch: stored bytes hash to {actual}"
            )

    def put_bytes(
        self,
        data: bytes | bytearray | memoryview,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        """Atomically persist bytes and return their immutable reference.

        Repeating the operation for identical bytes is idempotent.  If an
        existing path has been tampered with, the corruption is reported and
        the store never silently repairs or overwrites it.
        """

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        payload = bytes(data)
        ref = ArtifactRef(
            sha256=sha256_bytes(payload),
            size=len(payload),
            media_type=media_type,
        )
        destination = self.path_for(ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Persist a newly-created shard name before publishing an object in it.
        self._fsync_directory(self.objects_dir)

        if destination.exists():
            self._verify(destination.read_bytes(), ref)
            # This also closes the race with another writer that linked the same
            # object but has not yet flushed the directory entry.
            self._fsync_directory(destination.parent)
            return ref

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=".artifact-",
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)

            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                # Another writer won the same content-addressed race.  It must
                # still be verified rather than trusted based on its filename.
                self._verify(destination.read_bytes(), ref)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        # Flushing after temporary cleanup persists both publication of the
        # immutable name and removal of the staging name.
        self._fsync_directory(destination.parent)
        return ref

    def put_json(self, value: Any, *, media_type: str = "application/json") -> ArtifactRef:
        """Canonicalize and persist a JSON-compatible value."""

        return self.put_bytes(canonical_json_bytes(value), media_type=media_type)

    def get_bytes(self, ref_or_digest: ArtifactRef | str) -> bytes:
        """Read bytes only after verifying their digest and declared size."""

        digest = self._digest(ref_or_digest)
        path = self.path_for(digest)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(f"artifact {digest} was not found") from exc

        if isinstance(ref_or_digest, ArtifactRef):
            ref = ref_or_digest
        else:
            ref = ArtifactRef(
                sha256=digest,
                size=len(data),
                media_type="application/octet-stream",
            )
        self._verify(data, ref)
        return data

    def get_json(
        self,
        ref_or_digest: ArtifactRef | str,
        model_type: type[T] | None = None,
    ) -> Any | T:
        """Load verified JSON, optionally validating it as ``model_type``."""

        digest = self._digest(ref_or_digest)
        if isinstance(ref_or_digest, ArtifactRef) and not self._is_json_media_type(
            ref_or_digest.media_type
        ):
            raise ArtifactMediaTypeError(
                f"artifact {digest} declares non-JSON media type {ref_or_digest.media_type!r}"
            )

        data = self.get_bytes(ref_or_digest)

        def reject_non_finite(token: str) -> None:
            raise ValueError(f"non-finite JSON number is forbidden: {token}")

        def parse_finite_float(token: str) -> float:
            value = float(token)
            if not math.isfinite(value):
                raise ValueError(f"JSON number is outside the finite float range: {token}")
            return value

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate JSON object key is forbidden: {key!r}")
                value[key] = item
            return value

        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactDecodeError(f"artifact {digest} is not valid UTF-8 JSON") from exc
        try:
            value = json.loads(
                decoded,
                parse_constant=reject_non_finite,
                parse_float=parse_finite_float,
                object_pairs_hook=reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ArtifactDecodeError(f"artifact {digest} is not valid JSON: {exc}") from exc

        try:
            canonical = canonical_json_bytes(value)
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise ArtifactDecodeError(
                f"artifact {digest} cannot be represented as canonical JSON"
            ) from exc
        if canonical != data:
            raise ArtifactDecodeError(f"artifact {digest} JSON bytes are not in canonical form")

        if model_type is None:
            return value
        # Validate through Pydantic's JSON entry point.  Strict models may
        # correctly accept a JSON array for a tuple while rejecting a Python
        # list supplied directly by an untrusted caller.
        validated = TypeAdapter(model_type).validate_json(data)
        try:
            typed_canonical = canonical_json_bytes(validated)
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise ArtifactDecodeError(
                f"artifact {digest} typed value cannot be represented as canonical JSON"
            ) from exc
        if typed_canonical != data:
            raise ArtifactDecodeError(
                f"artifact {digest} bytes are not canonical for {model_type.__name__}"
            )
        return validated

    @staticmethod
    def _is_json_media_type(media_type: str) -> bool:
        """Return whether a declared media type denotes JSON syntax."""

        base_type = media_type.partition(";")[0].strip().lower()
        return base_type == "application/json" or base_type.endswith("+json")


__all__ = [
    "ArtifactDecodeError",
    "ArtifactIntegrityError",
    "ArtifactMediaTypeError",
    "ArtifactNotFoundError",
    "ArtifactStore",
    "ArtifactStoreError",
]
