"""Pinned, attributable benchmark dataset materialization.

Dataset bytes are deliberately not packaged in the wheel.  This module keeps
the small piece of trusted metadata needed to fetch them reproducibly and
materializes a file only after its exact byte length and SHA-256 digest match
the registry entry.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import urllib.request
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

GSM8K_UPSTREAM_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
DEFAULT_DATASET_DOWNLOAD_TIMEOUT_SECONDS = 60.0
GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
_GSM8K_RAW_ROOT = (
    "https://raw.githubusercontent.com/openai/grade-school-math/" + GSM8K_UPSTREAM_COMMIT
)


class DatasetIntegrityError(ValueError):
    """A dataset file did not match its frozen registry identity."""


class _ReadableResponse(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...

    def close(self) -> None: ...


DatasetOpener = Callable[[str], _ReadableResponse]


class DatasetArtifact(ImmutableModel):
    """One byte-exact upstream file in a dataset release."""

    schema_version: Literal["1"] = "1"
    role: NonEmptyStr
    filename: NonEmptyStr
    url: NonEmptyStr
    size: Annotated[int, Field(gt=0, strict=True)]
    sha256: Sha256
    media_type: Literal["application/jsonl"] = "application/jsonl"

    @field_validator("filename")
    @classmethod
    def filename_is_a_single_safe_component(cls, value: str) -> str:
        path = Path(value)
        if (
            path.name != value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "\x00" in value
        ):
            raise ValueError("dataset filename must be one safe path component")
        return value

    @field_validator("url")
    @classmethod
    def source_uses_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("dataset URL must use HTTPS")
        return value


class DatasetSnapshot(ImmutableModel):
    """One immutable byte snapshot and its self-verified content identity."""

    content: bytes = Field(repr=False)
    size: Annotated[int, Field(ge=0, strict=True)]
    sha256: Sha256

    @model_validator(mode="after")
    def identity_matches_content(self) -> DatasetSnapshot:
        if self.size != len(self.content):
            raise ValueError("dataset snapshot size does not match its content")
        if self.sha256 != hashlib.sha256(self.content).hexdigest():
            raise ValueError("dataset snapshot SHA-256 does not match its content")
        return self


class DatasetLicense(ImmutableModel):
    """Attribution and license provenance for one frozen dataset release."""

    spdx_id: NonEmptyStr
    name: NonEmptyStr
    url: NonEmptyStr
    copyright: NonEmptyStr

    @field_validator("url")
    @classmethod
    def license_uses_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("license URL must use HTTPS")
        return value


class DatasetProvenance(ImmutableModel):
    """Immutable registry entry for an attributable upstream snapshot."""

    schema_version: Literal["1"] = "1"
    dataset_id: NonEmptyStr
    display_name: NonEmptyStr
    upstream_repository: NonEmptyStr
    upstream_commit: GitCommit
    license: DatasetLicense
    artifacts: Annotated[tuple[DatasetArtifact, ...], Field(min_length=1)]

    @field_validator("upstream_repository")
    @classmethod
    def repository_uses_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("upstream repository URL must use HTTPS")
        return value.rstrip("/")

    @field_validator("artifacts")
    @classmethod
    def canonicalize_artifacts(
        cls,
        artifacts: tuple[DatasetArtifact, ...],
    ) -> tuple[DatasetArtifact, ...]:
        preferred_order = {"train": 0, "test": 1}
        return tuple(
            sorted(
                artifacts,
                key=lambda artifact: (preferred_order.get(artifact.role, 2), artifact.role),
            )
        )

    @model_validator(mode="after")
    def roles_and_filenames_are_unique(self) -> DatasetProvenance:
        roles = tuple(artifact.role for artifact in self.artifacts)
        filenames = tuple(artifact.filename for artifact in self.artifacts)
        if len(roles) != len(set(roles)):
            raise ValueError("dataset artifact roles must be unique")
        if len(filenames) != len(set(filenames)):
            raise ValueError("dataset artifact filenames must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        """Content identity of the complete provenance record."""

        return canonical_sha256(self)

    def artifact(self, role: str) -> DatasetArtifact:
        """Resolve a role without accepting an unregistered path or URL."""

        for artifact in self.artifacts:
            if artifact.role == role:
                return artifact
        raise KeyError(f"dataset {self.dataset_id!r} has no artifact role {role!r}")


class DatasetRegistry(ImmutableModel):
    """Frozen collection of byte-exact dataset releases."""

    schema_version: Literal["1"] = "1"
    datasets: Annotated[tuple[DatasetProvenance, ...], Field(min_length=1)]

    @field_validator("datasets")
    @classmethod
    def canonicalize_datasets(
        cls,
        datasets: tuple[DatasetProvenance, ...],
    ) -> tuple[DatasetProvenance, ...]:
        return tuple(sorted(datasets, key=lambda dataset: dataset.dataset_id))

    @model_validator(mode="after")
    def dataset_ids_are_unique(self) -> DatasetRegistry:
        dataset_ids = tuple(dataset.dataset_id for dataset in self.datasets)
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("dataset registry IDs must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        """Content identity of the registry, independent of input order."""

        return canonical_sha256(self)

    def get(self, dataset_id: str) -> DatasetProvenance:
        """Return one registered release or fail closed."""

        for dataset in self.datasets:
            if dataset.dataset_id == dataset_id:
                return dataset
        raise KeyError(f"unknown dataset ID: {dataset_id!r}")


GSM8K_PROVENANCE = DatasetProvenance(
    dataset_id="gsm8k@3101c7d5072418e28b9008a6636bde82a006892c",
    display_name="GSM8K (Grade School Math 8K)",
    upstream_repository="https://github.com/openai/grade-school-math",
    upstream_commit=GSM8K_UPSTREAM_COMMIT,
    license=DatasetLicense(
        spdx_id="MIT",
        name="MIT License",
        url=_GSM8K_RAW_ROOT + "/LICENSE",
        copyright="Copyright (c) 2021 OpenAI",
    ),
    artifacts=(
        DatasetArtifact(
            role="train",
            filename="train.jsonl",
            url=_GSM8K_RAW_ROOT + "/grade_school_math/data/train.jsonl",
            size=4_166_206,
            sha256="17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        ),
        DatasetArtifact(
            role="test",
            filename="test.jsonl",
            url=_GSM8K_RAW_ROOT + "/grade_school_math/data/test.jsonl",
            size=749_738,
            sha256="3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        ),
    ),
)

DATASET_REGISTRY = DatasetRegistry(datasets=(GSM8K_PROVENANCE,))


def read_dataset_snapshot(
    path: Path,
    *,
    artifact: DatasetArtifact | None = None,
    chunk_size: int = 1024 * 1024,
) -> DatasetSnapshot:
    """Read once, then bind all later consumers to the same immutable bytes.

    When an artifact is supplied, size and SHA-256 are checked against the
    registry entry before the snapshot is returned.  Callers that parse or
    grade dataset content must consume ``snapshot.content`` rather than reopen
    ``path`` after this function returns.
    """

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    checked_artifact = (
        None if artifact is None else DatasetArtifact.model_validate(artifact, strict=True)
    )
    path = Path(path)
    if path.is_symlink():
        raise DatasetIntegrityError(f"dataset target must not be a symbolic link: {path}")
    if not path.is_file():
        raise DatasetIntegrityError(f"dataset file does not exist: {path}")

    digest = hashlib.sha256()
    size = 0
    content = bytearray()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            size += len(chunk)
            if checked_artifact is not None and size > checked_artifact.size:
                raise DatasetIntegrityError(
                    f"dataset content exceeded registered size for {checked_artifact.role}"
                )
            digest.update(chunk)
            content.extend(chunk)

    actual_sha256 = digest.hexdigest()
    if checked_artifact is not None:
        if size != checked_artifact.size:
            raise DatasetIntegrityError(
                f"dataset size mismatch for {checked_artifact.role}: "
                f"expected {checked_artifact.size}, got {size}"
            )
        if actual_sha256 != checked_artifact.sha256:
            raise DatasetIntegrityError(
                f"dataset SHA-256 mismatch for {checked_artifact.role}: "
                f"expected {checked_artifact.sha256}, got {actual_sha256}"
            )
    return DatasetSnapshot(content=bytes(content), size=size, sha256=actual_sha256)


def verify_dataset_artifact(path: Path, artifact: DatasetArtifact) -> None:
    """Verify a local file against a registry artifact or raise."""

    read_dataset_snapshot(path, artifact=artifact)


def _existing_artifact_is_valid(path: Path, artifact: DatasetArtifact) -> bool:
    try:
        verify_dataset_artifact(path, artifact)
    except DatasetIntegrityError:
        return False
    return True


def download_dataset_artifact(
    artifact: DatasetArtifact,
    directory: Path,
    *,
    opener: DatasetOpener | None = None,
    chunk_size: int = 1024 * 1024,
    timeout_seconds: float = DEFAULT_DATASET_DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Materialize one registry artifact with verify-before-replace semantics.

    A valid existing target is reused without network access.  New bytes are
    streamed into a sibling temporary file, bounded by the expected size, and
    atomically replace the target only after exact size and digest checks pass.
    Consequently a failed or malicious response cannot overwrite a previously
    valid target.
    """

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    artifact = DatasetArtifact.model_validate(artifact, strict=True)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    target = directory / artifact.filename
    if target.is_symlink():
        raise DatasetIntegrityError(f"dataset target must not be a symbolic link: {target}")
    if _existing_artifact_is_valid(target, artifact):
        return target

    open_url = opener
    if open_url is None:
        open_url = cast(
            DatasetOpener,
            lambda url: urllib.request.urlopen(url, timeout=float(timeout_seconds)),
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=directory,
            prefix=f".{artifact.filename}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            received = 0
            with closing(open_url(artifact.url)) as response:
                while chunk := response.read(chunk_size):
                    if not isinstance(chunk, bytes):
                        raise DatasetIntegrityError("dataset response returned non-byte content")
                    received += len(chunk)
                    if received > artifact.size:
                        raise DatasetIntegrityError(
                            f"dataset response exceeded registered size for {artifact.role}"
                        )
                    digest.update(chunk)
                    temporary.write(chunk)

            actual_sha256 = digest.hexdigest()
            if received != artifact.size:
                raise DatasetIntegrityError(
                    f"dataset size mismatch for {artifact.role}: "
                    f"expected {artifact.size}, got {received}"
                )
            if actual_sha256 != artifact.sha256:
                raise DatasetIntegrityError(
                    f"dataset SHA-256 mismatch for {artifact.role}: "
                    f"expected {artifact.sha256}, got {actual_sha256}"
                )
            temporary.flush()
            os.fsync(temporary.fileno())

        os.replace(temporary_path, target)
        temporary_path = None
        return target
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def materialize_dataset(
    provenance: DatasetProvenance,
    directory: Path,
    *,
    opener: DatasetOpener | None = None,
) -> tuple[Path, ...]:
    """Download every registered artifact in canonical role order."""

    provenance = DatasetProvenance.model_validate(provenance, strict=True)
    return tuple(
        download_dataset_artifact(artifact, directory, opener=opener)
        for artifact in provenance.artifacts
    )


__all__ = [
    "DATASET_REGISTRY",
    "DEFAULT_DATASET_DOWNLOAD_TIMEOUT_SECONDS",
    "GSM8K_PROVENANCE",
    "GSM8K_UPSTREAM_COMMIT",
    "DatasetArtifact",
    "DatasetIntegrityError",
    "DatasetLicense",
    "DatasetOpener",
    "DatasetProvenance",
    "DatasetRegistry",
    "DatasetSnapshot",
    "GitCommit",
    "download_dataset_artifact",
    "materialize_dataset",
    "read_dataset_snapshot",
    "verify_dataset_artifact",
]
