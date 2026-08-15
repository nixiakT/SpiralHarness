#!/usr/bin/env python3
"""Pre-import BFCL development-environment integrity checks.

Run this script with ``-I -S -B``.  It uses only the standard library and
never imports ``bfcl_eval``.  These checks detect local drift; they are not a
signed supply-chain attestation, a syscall sandbox, or a network-isolation
attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
from collections.abc import Iterable
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

ENVIRONMENT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ENVIRONMENT_ROOT / "snapshot.json"
LOCK_PATH = ENVIRONMENT_ROOT / "uv.lock"

EXPECTED_REPOSITORY = "https://github.com/ShishirPatil/gorilla.git"
EXPECTED_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
EXPECTED_PACKAGE_ROOT = PurePosixPath("berkeley-function-call-leaderboard/bfcl_eval")
EXPECTED_ARCHIVE = (
    193,
    13_742_385,
    "d85faf8b05fc8ccc9624f31d0c9692f4512553ac9f0c2e241b25dde5a3ea866b",
)
EXPECTED_PACKAGE = (
    183,
    13_513_439,
    "3753addd78c10a6e59e3488ffdc5fb38cb46929380925ccfaecfcdb1d8b533b2",
)
EXPECTED_DISTRIBUTION = "bfcl-eval"
EXPECTED_VERSION = "2026.3.23"
EXPECTED_INDEX = "https://pypi.org/simple"
EXPECTED_WHEEL = (
    "bfcl_eval-2026.3.23-py3-none-any.whl",
    "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a",
)
EXPECTED_SDIST = (
    "bfcl_eval-2026.3.23.tar.gz",
    "4a3869673721fa59be93d8f55ca92d69ab5797058aed792149c6adafda064bc1",
)
EXPECTED_LOCK_SHA256 = "d8ab53a5381badb18ea63684bda79d17a8a46ca7b2b01d58a8af8ba1f5d9ffc3"
EXPECTED_PYTHON = (3, 12, 13)
EXPECTED_CONTROL_MODES = {
    ".python-version": "0644",
    "pyproject.toml": "0644",
    "self_test.py": "0755",
    "setup.sh": "0755",
    "smoke_evaluator.py": "0755",
    "verify.sh": "0755",
    "verify_snapshot.py": "0755",
}
EXPECTED_CONTROL_FILES = frozenset(EXPECTED_CONTROL_MODES)


class VerificationError(RuntimeError):
    """A development-environment integrity precondition failed."""


def _require_safe_interpreter_flags() -> None:
    if not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode:
        raise VerificationError("verifier requires Python flags -I -S -B")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise VerificationError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise VerificationError(f"{label} is not a regular, non-symlink file: {path}")


def _reject_symlink_ancestors(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise VerificationError(f"{label} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise VerificationError(f"{label} has a symlink component: {current}")


def _resolve_contained(path: Path, parent: Path, *, label: str) -> Path:
    """Resolve a path below a canonical parent and reject symlink components."""

    canonical_parent = parent.resolve(strict=True)
    if not path.is_absolute():
        raise VerificationError(f"{label} must be an absolute path")
    try:
        lexical_relative = path.relative_to(canonical_parent)
    except ValueError as exc:
        raise VerificationError(f"{label} is lexically outside {canonical_parent}") from exc
    current = canonical_parent
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            raise VerificationError(f"{label} contains a symlink component: {current}")
    try:
        canonical = path.resolve(strict=True)
        canonical.relative_to(canonical_parent)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise VerificationError(f"{label} is missing or outside {canonical_parent}") from exc
    return canonical


def _load_manifest() -> dict[str, Any]:
    _require_regular_file(MANIFEST_PATH, "snapshot manifest")
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError("snapshot manifest is not valid UTF-8 JSON") from exc
    upstream = payload.get("upstream", {})
    distribution = payload.get("distribution", {})
    attestations = payload.get("attestations", {})
    required = {
        "schema_version": "2",
        "repository": EXPECTED_REPOSITORY,
        "commit": EXPECTED_COMMIT,
        "package_root": str(EXPECTED_PACKAGE_ROOT),
        "distribution_name": EXPECTED_DISTRIBUTION,
        "distribution_version": EXPECTED_VERSION,
        "index": EXPECTED_INDEX,
        "dependency_environment_attested": False,
        "network_isolation_attested": False,
        "reportable_result": False,
    }
    observed = {
        "schema_version": payload.get("schema_version"),
        "repository": upstream.get("repository"),
        "commit": upstream.get("commit"),
        "package_root": upstream.get("package_root"),
        "distribution_name": distribution.get("name"),
        "distribution_version": distribution.get("version"),
        "index": distribution.get("index"),
        "dependency_environment_attested": attestations.get("dependency_environment_attested"),
        "network_isolation_attested": attestations.get("network_isolation_attested"),
        "reportable_result": attestations.get("reportable_result"),
    }
    if observed != required:
        raise VerificationError("snapshot identities or attestation boundaries differ")
    _verify_control_files(payload)
    return payload


def _verify_control_files(manifest: dict[str, Any]) -> str:
    declared = manifest.get("control_files")
    if not isinstance(declared, dict) or frozenset(declared) != EXPECTED_CONTROL_FILES:
        raise VerificationError("snapshot control-file roster differs")
    observed: dict[str, dict[str, str]] = {}
    for relative_name in sorted(EXPECTED_CONTROL_FILES):
        path = ENVIRONMENT_ROOT / relative_name
        _require_regular_file(path, f"control file {relative_name}")
        digest = _sha256(path)
        mode = f"{stat.S_IMODE(path.stat().st_mode):04o}"
        identity = {"sha256": digest, "mode": mode}
        if mode != EXPECTED_CONTROL_MODES[relative_name] or declared.get(relative_name) != identity:
            raise VerificationError(f"control file seal differs: {relative_name}")
        observed[relative_name] = identity
    canonical = json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _reject_bytecode(root: Path, *, scope: str) -> None:
    offenders: list[str] = []
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if "__pycache__" in relative.parts or candidate.suffix in {".pyc", ".pyo"}:
            offenders.append(relative.as_posix())
            if len(offenders) == 5:
                break
    if offenders:
        raise VerificationError(f"pre-existing Python bytecode in {scope}: {', '.join(offenders)}")


def _tree_files(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise VerificationError(f"tree root is missing or a symlink: {root}")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if candidate.is_symlink():
            raise VerificationError(f"tree contains symlink: {relative.as_posix()}")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise VerificationError(
                f"tree entry escaped or changed: {relative.as_posix()}"
            ) from exc
        info = candidate.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise VerificationError(f"tree contains non-regular entry: {relative.as_posix()}")
        files.append(candidate)
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def _tree_identity(
    root: Path,
    files: Iterable[Path],
    *,
    logical_prefix: PurePosixPath | None = None,
) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in files:
        relative = PurePosixPath(path.relative_to(root).as_posix())
        logical_path = relative if logical_prefix is None else logical_prefix / relative
        content = path.read_bytes()
        leaf = hashlib.sha256(content).hexdigest()
        digest.update(str(logical_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(leaf.encode("ascii"))
        digest.update(b"\n")
        total_bytes += len(content)
        file_count += 1
    return file_count, total_bytes, digest.hexdigest()


def _assert_identity(
    label: str,
    observed: tuple[int, int, str],
    expected: tuple[int, int, str],
) -> None:
    if observed != expected:
        raise VerificationError(f"{label} identity differs: {observed!r} != {expected!r}")


def _base_receipt(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "dependency_environment_attested": False,
        "network_isolation_attested": False,
        "reportable_result": False,
        "control_plane_sha256": _verify_control_files(manifest),
    }


def verify_source(root: Path) -> dict[str, Any]:
    """Verify a selected-path ``git archive`` from the pinned commit."""

    manifest = _load_manifest()
    root = _resolve_contained(root, ENVIRONMENT_ROOT, label="source archive root")
    if (root / ".git").exists():
        raise VerificationError("source must be a clean git archive, not a working checkout")
    _reject_bytecode(root, scope="source archive")

    archive_identity = _tree_identity(root, _tree_files(root))
    _assert_identity("source archive", archive_identity, EXPECTED_ARCHIVE)
    package_root = root.joinpath(*EXPECTED_PACKAGE_ROOT.parts)
    package_identity = _tree_identity(
        package_root,
        _tree_files(package_root),
        logical_prefix=EXPECTED_PACKAGE_ROOT,
    )
    _assert_identity("source package", package_identity, EXPECTED_PACKAGE)

    upstream = manifest["upstream"]
    declared_archive = upstream["archive_tree"]
    declared_package = upstream["package_tree"]
    if declared_archive != {
        "file_count": EXPECTED_ARCHIVE[0],
        "total_bytes": EXPECTED_ARCHIVE[1],
        "sha256": EXPECTED_ARCHIVE[2],
    } or declared_package != {
        "file_count": EXPECTED_PACKAGE[0],
        "total_bytes": EXPECTED_PACKAGE[1],
        "sha256": EXPECTED_PACKAGE[2],
    }:
        raise VerificationError("declared source identities differ from the verifier")
    return {
        **_base_receipt(manifest),
        "kind": "bfcl-v4-source-development-check",
        "commit": EXPECTED_COMMIT,
        "archive_file_count": archive_identity[0],
        "archive_sha256": archive_identity[2],
        "package_sha256": package_identity[2],
        "bytecode_found": False,
        "bfcl_imported": False,
        "source_snapshot_bound": True,
    }


def _site_packages(venv: Path) -> Path:
    candidate = venv / "lib/python3.12/site-packages"
    return _resolve_contained(candidate, venv, label="site-packages")


def _metadata_identity(dist_info: Path) -> tuple[str, str]:
    metadata_path = dist_info / "METADATA"
    _require_regular_file(metadata_path, "BFCL distribution metadata")
    metadata = BytesParser().parsebytes(metadata_path.read_bytes())
    return metadata.get("Name", ""), metadata.get("Version", "")


def _verify_lock(manifest: dict[str, Any]) -> None:
    _require_regular_file(LOCK_PATH, "uv lockfile")
    lock_sha256 = _sha256(LOCK_PATH)
    if lock_sha256 != EXPECTED_LOCK_SHA256:
        raise VerificationError(f"uv lockfile differs: {lock_sha256} != {EXPECTED_LOCK_SHA256}")
    if manifest.get("lock", {}).get("sha256") != EXPECTED_LOCK_SHA256:
        raise VerificationError("snapshot lock seal differs")
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - guarded by Python 3.12 check
        raise VerificationError("installed verification requires Python 3.12 tomllib") from exc
    with LOCK_PATH.open("rb") as stream:
        lock = tomllib.load(stream)
    matches = [
        item for item in lock.get("package", []) if item.get("name") == EXPECTED_DISTRIBUTION
    ]
    if len(matches) != 1:
        raise VerificationError("uv lockfile must contain exactly one bfcl-eval record")
    package = matches[0]
    if package.get("version") != EXPECTED_VERSION:
        raise VerificationError("uv lockfile has the wrong bfcl-eval version")
    if package.get("source") != {"registry": EXPECTED_INDEX}:
        raise VerificationError("uv lockfile has the wrong bfcl-eval source index")

    expected_wheel_name, expected_wheel_sha = EXPECTED_WHEEL
    expected_sdist_name, expected_sdist_sha = EXPECTED_SDIST
    wheels = package.get("wheels", [])
    if len(wheels) != 1:
        raise VerificationError("uv lockfile must list one bfcl-eval wheel candidate")
    wheel = wheels[0]
    sdist = package.get("sdist", {})
    if not wheel.get("url", "").endswith("/" + expected_wheel_name):
        raise VerificationError("uv lockfile has the wrong bfcl-eval wheel filename")
    if wheel.get("hash") != "sha256:" + expected_wheel_sha:
        raise VerificationError("uv lockfile has the wrong bfcl-eval wheel hash")
    if not sdist.get("url", "").endswith("/" + expected_sdist_name):
        raise VerificationError("uv lockfile has the wrong bfcl-eval sdist filename")
    if sdist.get("hash") != "sha256:" + expected_sdist_sha:
        raise VerificationError("uv lockfile has the wrong bfcl-eval sdist hash")


def _parse_pyvenv(venv: Path) -> dict[str, str]:
    path = venv / "pyvenv.cfg"
    _require_regular_file(path, "pyvenv.cfg")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            raise VerificationError("pyvenv.cfg contains a malformed line")
        name, value = raw_line.split("=", 1)
        values[name.strip()] = value.strip()
    return values


def _verify_runtime_observation(manifest: dict[str, Any], venv: Path) -> dict[str, Any]:
    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise VerificationError(
            f"wrong verifier interpreter: {sys.version_info[:3]!r} != {EXPECTED_PYTHON!r}"
        )
    expected_launcher = venv / "bin/python"
    if Path(sys.executable).absolute() != expected_launcher:
        raise VerificationError("installed verifier was not launched by the requested venv path")
    config = _parse_pyvenv(venv)
    if config.get("version_info") != ".".join(map(str, EXPECTED_PYTHON)):
        raise VerificationError("pyvenv.cfg has the wrong Python patch version")
    real_python = expected_launcher.resolve(strict=True)
    _reject_symlink_ancestors(real_python, label="resolved Python executable")
    runtime = manifest.get("runtime", {})
    expected_binary = runtime.get("python_executable_observation", {})
    observed = {"sha256": _sha256(real_python), "size": real_python.stat().st_size}
    if observed != expected_binary:
        raise VerificationError("Python executable observation differs")
    return observed


def verify_installed(venv: Path) -> dict[str, Any]:
    """Inspect the synchronized BFCL wheel tree without importing it."""

    manifest = _load_manifest()
    venv = _resolve_contained(venv, ENVIRONMENT_ROOT, label="evaluation virtualenv")
    runtime_observation = _verify_runtime_observation(manifest, venv)
    if sys.modules.get("bfcl_eval") is not None:
        raise VerificationError("bfcl_eval was imported before installed-tree verification")
    _reject_bytecode(venv, scope="evaluation virtualenv")
    site_packages = _site_packages(venv)

    package_root = _resolve_contained(
        site_packages / "bfcl_eval", site_packages, label="installed bfcl_eval"
    )
    package_identity = _tree_identity(
        package_root,
        _tree_files(package_root),
        logical_prefix=EXPECTED_PACKAGE_ROOT,
    )
    _assert_identity("installed bfcl-eval package", package_identity, EXPECTED_PACKAGE)

    dist_info_candidates = tuple(site_packages.glob("bfcl_eval-*.dist-info"))
    if len(dist_info_candidates) != 1:
        raise VerificationError("expected exactly one installed bfcl-eval dist-info directory")
    dist_info = _resolve_contained(
        dist_info_candidates[0], site_packages, label="bfcl-eval dist-info"
    )
    name, version = _metadata_identity(dist_info)
    if name.replace("_", "-").lower() != EXPECTED_DISTRIBUTION or version != EXPECTED_VERSION:
        raise VerificationError(f"installed distribution differs: {name!r} {version!r}")
    _verify_lock(manifest)
    return {
        **_base_receipt(manifest),
        "kind": "bfcl-v4-installed-development-check",
        "distribution": EXPECTED_DISTRIBUTION,
        "version": EXPECTED_VERSION,
        "python": ".".join(map(str, sys.version_info[:3])),
        "python_executable_observation": runtime_observation,
        "package_file_count": package_identity[0],
        "package_sha256": package_identity[2],
        "lock_sha256": EXPECTED_LOCK_SHA256,
        "bytecode_found": False,
        "bfcl_imported": False,
        "whole_environment_manifest_bound": False,
    }


def verify_tools(uv_executable: Path) -> dict[str, Any]:
    """Bind observed bootstrap executables without claiming their provenance."""

    manifest = _load_manifest()
    if not uv_executable.is_absolute():
        raise VerificationError("uv executable path must be absolute")
    if uv_executable.is_symlink():
        raise VerificationError("uv executable path must not be a symlink")
    _reject_symlink_ancestors(uv_executable, label="uv executable")
    _require_regular_file(uv_executable, "uv executable")
    canonical_uv = uv_executable.resolve(strict=True)
    runtime = manifest.get("runtime", {})
    expected_uv = runtime.get("uv_executable_observation", {})
    observed_uv = {"sha256": _sha256(canonical_uv), "size": canonical_uv.stat().st_size}
    if observed_uv != {key: expected_uv.get(key) for key in ("sha256", "size")}:
        raise VerificationError("uv executable observation differs")
    completed = subprocess.run(
        [str(canonical_uv), "--version"],
        check=False,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        text=True,
        timeout=10,
    )
    version_output = completed.stdout.strip()
    if completed.returncode != 0 or version_output != expected_uv.get("version_output"):
        raise VerificationError("uv version observation differs")

    bootstrap_python = Path(sys.executable).resolve(strict=True)
    _reject_symlink_ancestors(bootstrap_python, label="bootstrap Python executable")
    expected_bootstrap = runtime.get("bootstrap_python_executable_observation", {})
    observed_bootstrap = {
        "sha256": _sha256(bootstrap_python),
        "size": bootstrap_python.stat().st_size,
    }
    if observed_bootstrap != expected_bootstrap:
        raise VerificationError("bootstrap Python executable observation differs")
    return {
        **_base_receipt(manifest),
        "kind": "bfcl-v4-bootstrap-tool-observation",
        "uv": {**observed_uv, "version_output": version_output},
        "bootstrap_python": observed_bootstrap,
        "uv_executable_provenance_attested": False,
        "bootstrap_python_provenance_attested": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source", help="check a clean selected-path git archive")
    source.add_argument("--root", type=Path, required=True)
    installed = subparsers.add_parser("installed", help="check a synchronized virtualenv")
    installed.add_argument("--venv", type=Path, required=True)
    tools = subparsers.add_parser("tools", help="check observed bootstrap executables")
    tools.add_argument("--uv", type=Path, required=True)
    return parser


def main() -> int:
    try:
        _require_safe_interpreter_flags()
        args = _parser().parse_args()
        if args.command == "source":
            result = verify_source(args.root)
        elif args.command == "installed":
            result = verify_installed(args.venv)
        else:
            result = verify_tools(args.uv)
    except (OSError, subprocess.SubprocessError, VerificationError) as exc:
        print(f"BFCL development check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
