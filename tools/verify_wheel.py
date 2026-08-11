#!/usr/bin/env python3
"""Verify that the built wheel works without access to the source checkout."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from email.parser import Parser
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_NAME = "spiral-harness"
PACKAGE_NAME = "spiral_harness"
SMOKE_PREFIX = "SPIRAL_WHEEL_SMOKE="


class WheelVerificationError(RuntimeError):
    """Raised when a wheel is unsafe, incomplete, or source-tree dependent."""


def _resolve_wheel(candidate: Path) -> Path:
    resolved = candidate.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix != ".whl":
            raise WheelVerificationError(f"expected a .whl file, got {resolved}")
        return resolved
    if not resolved.is_dir():
        raise WheelVerificationError(f"wheel path does not exist: {resolved}")
    wheels = sorted(resolved.glob("*.whl"))
    if len(wheels) != 1:
        rendered = ", ".join(path.name for path in wheels) or "none"
        raise WheelVerificationError(f"expected exactly one wheel in {resolved}; found {rendered}")
    return wheels[0].resolve()


def _normalized_distribution_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


def _verify_wheel_archive(wheel: Path) -> str:
    try:
        archive = zipfile.ZipFile(wheel)
    except (OSError, zipfile.BadZipFile) as exc:
        raise WheelVerificationError(f"cannot open wheel archive {wheel}: {exc}") from exc

    with archive:
        names = [item.filename for item in archive.infolist()]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise WheelVerificationError(
                "wheel contains duplicate archive members: " + ", ".join(duplicates)
            )

        unsafe = []
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                unsafe.append(name)
        if unsafe:
            raise WheelVerificationError(
                "wheel contains unsafe member paths: " + ", ".join(sorted(unsafe))
            )

        package_prefix = f"{PACKAGE_NAME}/"
        package_entries = [name for name in names if name.startswith(package_prefix)]
        if not package_entries:
            raise WheelVerificationError(f"wheel does not contain {package_prefix}")
        if names.count(f"{PACKAGE_NAME}/__init__.py") != 1:
            raise WheelVerificationError(
                f"wheel must contain exactly one {PACKAGE_NAME}/__init__.py"
            )

        forbidden_prefixes = (".github/", "docs/", "src/", "tests/", "tools/")
        forbidden = sorted(
            name for name in names if any(name.startswith(prefix) for prefix in forbidden_prefixes)
        )
        if forbidden:
            raise WheelVerificationError(
                "wheel contains repository-only content: " + ", ".join(forbidden)
            )

        python_outside_package = sorted(
            name
            for name in names
            if name.endswith((".py", ".pyi")) and not name.startswith(package_prefix)
        )
        if python_outside_package:
            raise WheelVerificationError(
                "wheel contains a second Python source tree: " + ", ".join(python_outside_package)
            )

        metadata_members = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_members) != 1:
            raise WheelVerificationError(
                f"wheel must contain exactly one METADATA file; found {len(metadata_members)}"
            )
        metadata = Parser().parsestr(archive.read(metadata_members[0]).decode("utf-8"))
        metadata_name = metadata.get("Name")
        metadata_version = metadata.get("Version")
        if metadata_name is None or _normalized_distribution_name(metadata_name) != DIST_NAME:
            raise WheelVerificationError(
                f"wheel distribution name is {metadata_name!r}, expected {DIST_NAME!r}"
            )
        if not metadata_version:
            raise WheelVerificationError("wheel METADATA has no Version")

        entry_point_members = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(entry_point_members) != 1:
            raise WheelVerificationError(
                "wheel must contain exactly one dist-info/entry_points.txt"
            )
        entry_points = configparser.ConfigParser(interpolation=None)
        entry_points.read_string(archive.read(entry_point_members[0]).decode("utf-8"))
        console_scripts = (
            entry_points["console_scripts"] if entry_points.has_section("console_scripts") else {}
        )
        if console_scripts.get("spiral") != "spiral_harness.cli:app":
            raise WheelVerificationError("wheel does not declare spiral = spiral_harness.cli:app")

    return metadata_version


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise WheelVerificationError(
            f"command failed ({completed.returncode}): {rendered}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _venv_spiral(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "spiral.exe"
    return venv / "bin" / "spiral"


def _parse_smoke_payload(output: str) -> dict[str, object]:
    payloads = [
        line.removeprefix(SMOKE_PREFIX)
        for line in output.splitlines()
        if line.startswith(SMOKE_PREFIX)
    ]
    if len(payloads) != 1:
        raise WheelVerificationError(
            f"isolated import emitted {len(payloads)} smoke payloads, expected one\n{output}"
        )
    try:
        payload = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise WheelVerificationError(f"isolated import emitted invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WheelVerificationError("isolated import payload is not a JSON object")
    return payload


def _as_path(payload: dict[str, object], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str):
        raise WheelVerificationError(f"isolated import payload has no string {key!r}")
    return Path(value).resolve()


def _verify_isolated_install(wheel: Path, *, metadata_version: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise WheelVerificationError("uv is required for isolated wheel verification")
    environment = _clean_environment()

    with tempfile.TemporaryDirectory(prefix="spiral-harness-wheel-") as temporary:
        temporary_root = Path(temporary).resolve()
        if temporary_root == PROJECT_ROOT or temporary_root.is_relative_to(PROJECT_ROOT):
            raise WheelVerificationError("temporary environment was created inside the checkout")

        venv = temporary_root / "venv"
        workdir = temporary_root / "outside-checkout"
        workdir.mkdir()
        _run(
            [uv, "venv", "--python", sys.executable, str(venv)],
            cwd=temporary_root,
            environment=environment,
        )
        python = _venv_python(venv)
        _run(
            [uv, "pip", "install", "--python", str(python), str(wheel)],
            cwd=temporary_root,
            environment=environment,
        )

        script = "\n".join(
            (
                "import json",
                "import os",
                "import sys",
                "from importlib.metadata import version",
                "from pathlib import Path",
                "import spiral_harness",
                "import spiral_harness.cli",
                "import spiral_harness.evolution.models",
                "payload = {",
                "    'cwd': str(Path.cwd().resolve()),",
                "    'distribution_version': version('spiral-harness'),",
                "    'package_file': str(Path(spiral_harness.__file__).resolve()),",
                "    'package_version': spiral_harness.__version__,",
                "    'prefix': str(Path(sys.prefix).resolve()),",
                "    'sys_path': [str(Path(item).resolve()) for item in sys.path if item],",
                "}",
                f"print('{SMOKE_PREFIX}' + json.dumps(payload, sort_keys=True))",
            )
        )
        imported = _run(
            [str(python), "-I", "-B", "-c", script],
            cwd=workdir,
            environment=environment,
        )
        payload = _parse_smoke_payload(imported.stdout)

        package_file = _as_path(payload, "package_file")
        prefix = _as_path(payload, "prefix")
        imported_cwd = _as_path(payload, "cwd")
        if not package_file.is_relative_to(venv):
            raise WheelVerificationError(
                f"installed package was not imported from the isolated venv: {package_file}"
            )
        if package_file.is_relative_to(PROJECT_ROOT):
            raise WheelVerificationError(f"installed package leaked from checkout: {package_file}")
        if prefix != venv and not prefix.is_relative_to(venv):
            raise WheelVerificationError(f"isolated interpreter has unexpected prefix: {prefix}")
        if imported_cwd != workdir:
            raise WheelVerificationError(
                f"isolated import ran from {imported_cwd}, expected {workdir}"
            )

        sys_path = payload.get("sys_path")
        if not isinstance(sys_path, list) or not all(isinstance(item, str) for item in sys_path):
            raise WheelVerificationError("isolated import payload has an invalid sys_path")
        checkout_paths = sorted(
            item
            for item in sys_path
            if Path(item).resolve() == PROJECT_ROOT
            or Path(item).resolve().is_relative_to(PROJECT_ROOT)
        )
        if checkout_paths:
            raise WheelVerificationError(
                "isolated interpreter can see checkout paths: " + ", ".join(checkout_paths)
            )

        distribution_version = payload.get("distribution_version")
        package_version = payload.get("package_version")
        if distribution_version != metadata_version or package_version != metadata_version:
            raise WheelVerificationError(
                "version mismatch: "
                f"METADATA={metadata_version!r}, distribution={distribution_version!r}, "
                f"package={package_version!r}"
            )

        spiral = _venv_spiral(venv)
        if not spiral.is_file():
            raise WheelVerificationError(f"installed console script is missing: {spiral}")
        help_result = _run(
            [str(spiral), "--help"],
            cwd=workdir,
            environment=environment,
        )
        if "usage" not in (help_result.stdout + help_result.stderr).casefold():
            raise WheelVerificationError("spiral --help did not print a usage message")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wheel",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "dist",
        help="wheel file or directory containing exactly one wheel (default: dist)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        wheel = _resolve_wheel(arguments.wheel)
        metadata_version = _verify_wheel_archive(wheel)
        _verify_isolated_install(wheel, metadata_version=metadata_version)
    except WheelVerificationError as exc:
        print(f"wheel verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"verified isolated wheel: {wheel.name} ({metadata_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
