from __future__ import annotations

import platform
import sys

import pytest

from spiral_harness.execution.runtime_identity import current_python_runtime_identity


def test_runtime_identity_records_the_actual_interpreter_and_platform() -> None:
    identity = current_python_runtime_identity(component="fixture-runtime", revision="v1")

    assert identity.startswith(
        f"fixture-runtime/{platform.python_implementation().casefold()}-"
        f"{platform.python_version()}-{sys.implementation.cache_tag}/"
    )
    assert f"/{platform.system().casefold()}-{platform.machine().casefold()}@v1" in identity
    assert "py3.12" not in identity


@pytest.mark.parametrize(
    ("component", "revision"),
    [("", "v1"), (" fixture", "v1"), ("fixture", ""), ("fixture", "v1 ")],
)
def test_runtime_identity_rejects_ambiguous_coordinates(
    component: str,
    revision: str,
) -> None:
    with pytest.raises(ValueError, match="exact non-empty"):
        current_python_runtime_identity(component=component, revision=revision)
