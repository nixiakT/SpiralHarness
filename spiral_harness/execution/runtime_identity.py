"""Credential-free identity for the local Python execution runtime."""

from __future__ import annotations

import platform
import sys


def current_python_runtime_identity(*, component: str, revision: str) -> str:
    """Return a host-agnostic but interpreter/platform-specific runtime identity."""

    for label, value in (("component", component), ("revision", revision)):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{label} must be an exact non-empty string")
    coordinates = {
        "implementation": platform.python_implementation().casefold(),
        "python_version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag or "",
        "system": platform.system().casefold(),
        "machine": platform.machine().casefold(),
    }
    missing = tuple(name for name, value in coordinates.items() if not value)
    if missing:
        raise RuntimeError(f"runtime identity coordinates unavailable: {', '.join(missing)}")
    return (
        f"{component}/{coordinates['implementation']}-{coordinates['python_version']}-"
        f"{coordinates['cache_tag']}/{coordinates['system']}-{coordinates['machine']}@{revision}"
    )


__all__ = ["current_python_runtime_identity"]
