from __future__ import annotations

import ast
from pathlib import Path


def test_native_adapter_template_is_valid_and_contains_no_credentials() -> None:
    path = Path("benchmarks/meta_harness_native/spiral_native.py")
    source = path.read_text(encoding="utf-8")

    ast.parse(source)
    assert "SPIRAL_META_HARNESS_MODE" in source
    assert "sk-" not in source
    assert "api_key" not in source.lower()
