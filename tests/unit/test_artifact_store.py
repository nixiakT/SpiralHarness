from __future__ import annotations

import json

import pytest

from spiral_harness.core import (
    ArtifactRef,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    canonical_json_bytes,
    sha256_bytes,
)
from spiral_harness.storage import (
    ArtifactDecodeError,
    ArtifactIntegrityError,
    ArtifactMediaTypeError,
    ArtifactNotFoundError,
    ArtifactStore,
)


def test_bytes_round_trip_and_duplicate_put_is_idempotent(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    payload = b"immutable trajectory bytes"

    first = store.put_bytes(payload, media_type="application/x-trajectory")
    second = store.put_bytes(payload, media_type="application/x-trajectory")

    assert first == second
    assert store.get_bytes(first) == payload
    assert store.get_bytes(first.sha256) == payload
    assert store.path_for(first).is_file()
    assert [path for path in store.objects_dir.rglob("*") if path.is_file()] == [
        store.path_for(first)
    ]


def test_json_put_is_canonical_and_can_return_a_validated_model(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    left = store.put_json({"z": 1, "a": {"y": 2, "b": 3}})
    right = store.put_json({"a": {"b": 3, "y": 2}, "z": 1})

    assert left == right
    assert store.get_bytes(left) == b'{"a":{"b":3,"y":2},"z":1}'
    assert store.get_json(left) == {"a": {"b": 3, "y": 2}, "z": 1}

    component = HarnessComponentRef(
        name="system",
        kind=ComponentKind.PROMPT,
        artifact=left,
    )
    component_ref = store.put_json(component)
    loaded_component = store.get_json(component_ref, HarnessComponentRef)

    assert loaded_component == component
    assert component_ref.sha256 == sha256_bytes(canonical_json_bytes(loaded_component))


@pytest.mark.parametrize("variant", ["alias", "missing-default", "unsorted", "whitespace"])
def test_typed_json_rejects_noncanonical_model_representations(tmp_path, variant: str) -> None:
    store = ArtifactStore(tmp_path)
    manifest = HarnessManifest(
        model_fingerprint="model",
        runtime_fingerprint="runtime",
        trusted_plane_version="trusted-plane-v1",
        parent=ArtifactRef(sha256="f" * 64, size=1, media_type="application/json"),
        components=(
            HarnessComponentRef(
                name="alpha",
                kind=ComponentKind.PROMPT,
                artifact=ArtifactRef(sha256="a" * 64, size=1, media_type="text/plain"),
            ),
            HarnessComponentRef(
                name="zeta",
                kind=ComponentKind.SKILL,
                artifact=ArtifactRef(sha256="b" * 64, size=1, media_type="text/plain"),
            ),
        ),
    )
    payload = manifest.model_dump(mode="json", by_alias=False, exclude_none=False)

    if variant == "alias":
        payload["parent_ref"] = payload.pop("parent")
    elif variant == "missing-default":
        del payload["schema_version"]
    elif variant == "unsorted":
        payload["components"].reverse()
    else:
        payload["model_fingerprint"] = " model "

    ref = store.put_json(payload)

    with pytest.raises(ArtifactDecodeError, match="not canonical for HarnessManifest"):
        store.get_json(ref, HarnessManifest)


def test_read_detects_same_size_hash_tampering(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"original")
    store.path_for(ref).write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        store.get_bytes(ref)
    with pytest.raises(ArtifactIntegrityError):
        store.put_bytes(b"original")


def test_read_detects_declared_size_mismatch(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"content")
    wrong_size = ArtifactRef(
        sha256=ref.sha256,
        size=ref.size + 1,
        media_type=ref.media_type,
    )

    with pytest.raises(ArtifactIntegrityError, match="size mismatch"):
        store.get_bytes(wrong_size)


def test_missing_artifact_has_a_domain_error(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    missing = ArtifactRef(sha256="0" * 64, size=0, media_type="application/octet-stream")

    with pytest.raises(ArtifactNotFoundError):
        store.get_bytes(missing)


def test_json_rejects_nan_on_write_and_decode(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.put_json({"score": float("nan")})

    # Bytes can represent arbitrary media, so a caller can store malformed JSON;
    # interpreting those bytes as JSON still fails closed.
    malformed = store.put_bytes(b'{"score":NaN}', media_type="application/json")
    with pytest.raises(ArtifactDecodeError):
        store.get_json(malformed)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"score":1e400}',
        b'{"score":1,"score":2}',
    ],
)
def test_json_decode_rejects_overflow_and_duplicate_keys(tmp_path, payload: bytes) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(payload, media_type="application/json")

    with pytest.raises(ArtifactDecodeError, match="not valid JSON"):
        store.get_json(ref)


def test_json_decode_requires_canonical_bytes(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b'{ "score": 1 }', media_type="application/json")

    with pytest.raises(ArtifactDecodeError, match="not in canonical form"):
        store.get_json(ref)


def test_typed_json_read_checks_declared_media_type(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    plain_text_ref = store.put_bytes(b"{}", media_type="text/plain")

    with pytest.raises(ArtifactMediaTypeError, match="non-JSON media type"):
        store.get_json(plain_text_ref)

    vendor_ref = store.put_json(
        {"schema_version": "1"},
        media_type="application/vnd.spiral-harness.manifest+json; charset=utf-8",
    )
    assert store.get_json(vendor_ref) == {"schema_version": "1"}


def test_invalid_json_has_a_decode_error(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"not json", media_type="application/json")

    with pytest.raises(ArtifactDecodeError):
        store.get_json(ref)


def test_artifact_ref_matches_standard_sha256(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    payload = {"score": 1}
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ref = store.put_json(payload)

    import hashlib

    assert ref.sha256 == hashlib.sha256(encoded).hexdigest()
    assert ref.size == len(encoded)
    assert ref.media_type == "application/json"


def test_relative_store_root_does_not_change_with_working_directory(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    store = ArtifactStore("relative-store")
    ref = store.put_bytes(b"stable root")

    monkeypatch.chdir(second)

    assert store.root == (first / "relative-store").resolve()
    assert store.get_bytes(ref) == b"stable root"


def test_successful_put_flushes_the_shard_directory(tmp_path, monkeypatch) -> None:
    store = ArtifactStore(tmp_path)
    synced_directories = []
    monkeypatch.setattr(store, "_fsync_directory", synced_directories.append)

    ref = store.put_bytes(b"durable directory entry")

    assert store.path_for(ref).parent in synced_directories
