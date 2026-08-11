from __future__ import annotations

import hashlib
import io
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from spiral_harness import benchmark as benchmark_package
from spiral_harness.benchmark.datasets import (
    DATASET_REGISTRY,
    GSM8K_PROVENANCE,
    GSM8K_UPSTREAM_COMMIT,
    DatasetArtifact,
    DatasetIntegrityError,
    DatasetLicense,
    DatasetOpener,
    DatasetProvenance,
    DatasetRegistry,
    DatasetSnapshot,
    GitCommit,
    download_dataset_artifact,
    materialize_dataset,
    read_dataset_snapshot,
    verify_dataset_artifact,
)


def test_dataset_and_adapter_public_types_are_available_from_package_facade() -> None:
    assert benchmark_package.DatasetOpener is DatasetOpener
    assert benchmark_package.DatasetSnapshot is DatasetSnapshot
    assert benchmark_package.GitCommit is GitCommit
    assert benchmark_package.GSM8K_ANSWER_RE.pattern == benchmark_package.GSM8K_ANSWER_PATTERN
    assert benchmark_package.GSM8KExecution.__name__ == "GSM8KExecution"


def artifact_for(
    content: bytes,
    *,
    role: str = "fixture",
    filename: str = "fixture.jsonl",
) -> DatasetArtifact:
    return DatasetArtifact(
        role=role,
        filename=filename,
        url=f"https://fixtures.invalid/{filename}",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def fixture_provenance(*artifacts: DatasetArtifact) -> DatasetProvenance:
    return DatasetProvenance(
        dataset_id="fixture@0123456789abcdef0123456789abcdef01234567",
        display_name="Fixture",
        upstream_repository="https://fixtures.invalid/repository",
        upstream_commit="0123456789abcdef0123456789abcdef01234567",
        license=DatasetLicense(
            spdx_id="MIT",
            name="MIT License",
            url="https://fixtures.invalid/LICENSE",
            copyright="Copyright Fixture",
        ),
        artifacts=artifacts,
    )


def test_gsm8k_registry_freezes_exact_upstream_snapshot_and_attribution() -> None:
    dataset = DATASET_REGISTRY.get(GSM8K_PROVENANCE.dataset_id)

    assert dataset == GSM8K_PROVENANCE
    assert dataset.upstream_commit == GSM8K_UPSTREAM_COMMIT
    assert dataset.upstream_repository == "https://github.com/openai/grade-school-math"
    assert dataset.license.spdx_id == "MIT"
    assert dataset.license.copyright == "Copyright (c) 2021 OpenAI"
    assert dataset.artifact("train").model_dump() == {
        "schema_version": "1",
        "role": "train",
        "filename": "train.jsonl",
        "url": (
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            f"{GSM8K_UPSTREAM_COMMIT}/grade_school_math/data/train.jsonl"
        ),
        "size": 4_166_206,
        "sha256": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        "media_type": "application/jsonl",
    }
    assert dataset.artifact("test").size == 749_738
    assert (
        dataset.artifact("test").sha256
        == "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
    )


def test_registry_contract_is_strict_canonical_and_immutable() -> None:
    first = artifact_for(b"one", role="z", filename="z.jsonl")
    second = artifact_for(b"two", role="a", filename="a.jsonl")
    provenance = fixture_provenance(first, second)
    registry = DatasetRegistry(datasets=(GSM8K_PROVENANCE, provenance))

    assert [item.role for item in provenance.artifacts] == ["a", "z"]
    assert (
        registry.fingerprint
        == DatasetRegistry(datasets=tuple(reversed(registry.datasets))).fingerprint
    )
    with pytest.raises((ValidationError, FrozenInstanceError)):
        provenance.dataset_id = "changed"
    for filename in ("../bad.jsonl", r"..\bad.jsonl"):
        with pytest.raises(ValidationError):
            DatasetArtifact(
                role="bad",
                filename=filename,
                url="http://fixtures.invalid/bad.jsonl",
                size=1,
                sha256="a" * 64,
            )
    with pytest.raises(ValidationError):
        DatasetProvenance(
            **{
                **provenance.model_dump(),
                "artifacts": (first, first),
            }
        )
    with pytest.raises(ValidationError):
        DatasetRegistry(datasets=(provenance, provenance))
    with pytest.raises(KeyError):
        registry.get("missing")
    with pytest.raises(KeyError):
        provenance.artifact("missing")


def test_downloader_streams_valid_bytes_then_atomically_materializes(tmp_path: Path) -> None:
    content = b'{"question":"q","answer":"#### 1"}\n'
    artifact = artifact_for(content)
    calls: list[str] = []

    def opener(url: str) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(content)

    target = download_dataset_artifact(artifact, tmp_path / "data", opener=opener, chunk_size=3)

    assert target == tmp_path / "data" / artifact.filename
    assert target.read_bytes() == content
    assert calls == [artifact.url]
    assert not list(target.parent.glob(".*.tmp"))
    verify_dataset_artifact(target, artifact)


def test_valid_existing_artifact_is_reused_without_opening_network(tmp_path: Path) -> None:
    content = b"registered bytes"
    artifact = artifact_for(content)
    target = tmp_path / artifact.filename
    target.write_bytes(content)

    def forbidden_opener(url: str) -> io.BytesIO:
        raise AssertionError(f"network must not be opened for valid target: {url}")

    assert download_dataset_artifact(artifact, tmp_path, opener=forbidden_opener) == target
    assert target.read_bytes() == content


def test_verified_snapshot_remains_bound_when_path_is_replaced(tmp_path: Path) -> None:
    content = b"registered snapshot"
    replacement = b"tampered after verification"
    artifact = artifact_for(content)
    target = tmp_path / artifact.filename
    target.write_bytes(content)

    snapshot = read_dataset_snapshot(target, artifact=artifact, chunk_size=2)
    target.write_bytes(replacement)

    assert snapshot.content == content
    assert snapshot.size == len(content)
    assert snapshot.sha256 == hashlib.sha256(content).hexdigest()
    with pytest.raises(DatasetIntegrityError):
        verify_dataset_artifact(target, artifact)


def test_dataset_snapshot_rejects_an_identity_not_derived_from_its_bytes() -> None:
    content = b"snapshot"
    sha256 = hashlib.sha256(content).hexdigest()

    with pytest.raises(ValidationError, match="size does not match"):
        DatasetSnapshot(content=content, size=len(content) + 1, sha256=sha256)
    with pytest.raises(ValidationError, match="SHA-256 does not match"):
        DatasetSnapshot(content=content, size=len(content), sha256="0" * 64)


@pytest.mark.parametrize("response", [b"short", b"x" * 26, b"x" * 100])
def test_invalid_response_never_replaces_existing_file(tmp_path: Path, response: bytes) -> None:
    expected = b"correct registered content"
    artifact = artifact_for(expected)
    target = tmp_path / artifact.filename
    preexisting = b"preexisting unverified bytes"
    target.write_bytes(preexisting)

    with pytest.raises(DatasetIntegrityError):
        download_dataset_artifact(
            artifact,
            tmp_path,
            opener=lambda _: io.BytesIO(response),
            chunk_size=2,
        )

    assert target.read_bytes() == preexisting
    assert not list(tmp_path.glob(".*.tmp"))


def test_downloader_rejects_non_byte_response_and_bad_chunk_size(tmp_path: Path) -> None:
    artifact = artifact_for(b"abc")

    class TextResponse:
        def read(self, size: int = -1) -> str:
            del size
            return "abc"

        def close(self) -> None:
            pass

    with pytest.raises(DatasetIntegrityError, match="non-byte"):
        download_dataset_artifact(artifact, tmp_path, opener=lambda _: TextResponse())  # type: ignore[arg-type,return-value]
    for chunk_size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            download_dataset_artifact(
                artifact,
                tmp_path,
                opener=lambda _: io.BytesIO(b"abc"),
                chunk_size=chunk_size,  # type: ignore[arg-type]
            )
    for timeout_seconds in (0.0, -1.0, True, float("inf"), float("nan"), "60"):
        with pytest.raises(ValueError, match="positive finite"):
            download_dataset_artifact(
                artifact,
                tmp_path,
                opener=lambda _: io.BytesIO(b"abc"),
                timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
            )


def test_verify_rejects_missing_wrong_and_symbolic_link_targets(tmp_path: Path) -> None:
    artifact = artifact_for(b"correct")
    target = tmp_path / artifact.filename
    with pytest.raises(DatasetIntegrityError, match="does not exist"):
        verify_dataset_artifact(target, artifact)

    target.write_bytes(b"wrong")
    with pytest.raises(DatasetIntegrityError, match="size mismatch"):
        verify_dataset_artifact(target, artifact)

    target.unlink()
    backing = tmp_path / "backing.jsonl"
    backing.write_bytes(b"correct")
    try:
        target.symlink_to(backing)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links are unavailable in this test environment: {error}")
    with pytest.raises(DatasetIntegrityError, match="symbolic link"):
        verify_dataset_artifact(target, artifact)
    with pytest.raises(DatasetIntegrityError, match="symbolic link"):
        download_dataset_artifact(artifact, tmp_path, opener=lambda _: io.BytesIO(b"correct"))


def test_materialize_dataset_uses_canonical_artifact_order(tmp_path: Path) -> None:
    a_content = b"a"
    z_content = b"z"
    provenance = fixture_provenance(
        artifact_for(z_content, role="z", filename="z.jsonl"),
        artifact_for(a_content, role="a", filename="a.jsonl"),
    )
    content_by_url = {artifact.url: artifact.role.encode() for artifact in provenance.artifacts}

    paths = materialize_dataset(
        provenance,
        tmp_path,
        opener=lambda url: io.BytesIO(content_by_url[url]),
    )

    assert [path.name for path in paths] == ["a.jsonl", "z.jsonl"]
