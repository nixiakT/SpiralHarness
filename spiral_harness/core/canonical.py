"""Deterministic JSON encoding used by the content-addressed data plane.

This is intentionally a small canonicalization contract rather than a claim of
full RFC 8785 compatibility.  SpiralHarness needs two guarantees at M0: object
key order cannot affect an artifact identity, and non-finite JSON numbers can
never enter the ledger.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import textwrap
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from pydantic import BaseModel


def _reject_non_string_mapping_keys(value: Any) -> None:
    """Reject mappings that JSON would silently coerce into string-keyed objects."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            _reject_non_string_mapping_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_string_mapping_keys(item)


def _normalize_json_value(value: Any) -> Any:
    """Recursively enforce the M0 JSON value and numeric identity contract."""

    if isinstance(value, BaseModel):
        return _json_value(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalized[key] = _normalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are forbidden in canonical JSON")
        # IEEE-754 negative zero has no distinct budget or score meaning here.
        # Normalizing it prevents two hashes for the same logical numeric value.
        if value == 0.0:
            return 0.0
    return value


def _json_value(value: Any) -> Any:
    """Convert a Pydantic object to its JSON representation.

    ``mode="json"`` is important here: it gives Pydantic-owned scalar types a
    documented JSON representation before the standard library encoder sees
    them.  Plain inputs are left alone so unsupported Python values fail closed
    instead of acquiring a surprising, process-dependent representation.
    """

    if isinstance(value, BaseModel):
        # ``model_copy(update=...)`` and ``model_construct(...)`` deliberately
        # bypass Pydantic validation.  Never let such an instance cross the
        # hashing boundary based only on its Python type.
        python_content = value.model_dump(
            mode="python",
            by_alias=False,
            exclude_none=False,
            round_trip=True,
            warnings="none",
        )
        _reject_non_string_mapping_keys(python_content)
        validated = type(value).model_validate(
            python_content,
            strict=True,
            by_name=True,
        )
        value = validated.model_dump(mode="json", by_alias=False, exclude_none=False)
    return _normalize_json_value(value)


def canonical_json(value: Any) -> str:
    """Return compact, deterministic JSON for a JSON-compatible value.

    Mapping keys are sorted recursively by :func:`json.dumps`.  NaN and both
    infinities raise :class:`ValueError`, because their spellings are not valid
    JSON and would make cross-runtime artifact identities unreliable.
    """

    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the UTF-8 bytes of :func:`canonical_json`."""

    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA-256 digest for *data*."""

    return hashlib.sha256(data).hexdigest()


def _normalized_python_source(value: object) -> bytes:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise ValueError("Python source is unavailable") from error
    normalized = textwrap.dedent(source).replace("\r\n", "\n").replace("\r", "\n")
    return (normalized.rstrip("\n") + "\n").encode("utf-8")


def callable_source_sha256(function: Callable[..., object]) -> str:
    """Hash canonical source text, never interpreter-specific bytecode.

    CPython bytecode is neither a complete implementation identity nor stable
    across supported Python versions.  Source fingerprints normalize indentation
    and line endings, then require a recoverable Python source definition.  This
    helper is intended for module-level Python callables shipped with the source
    distribution; dynamically generated functions fail closed.
    """

    if not inspect.isfunction(function):
        raise TypeError("source fingerprint requires a Python function")
    return sha256_bytes(_normalized_python_source(function))


def module_source_sha256(module: ModuleType) -> str:
    """Hash a module's normalized source as an implementation-bundle identity.

    A function-only hash omits module constants and local helpers that can alter
    its behavior.  Module-level source is the smallest fail-closed bundle for
    benchmark adapters whose prompt builders depend on that surrounding state.
    Built-in, namespace, and bytecode-only modules fail closed.
    """

    if not inspect.ismodule(module):
        raise TypeError("source fingerprint requires a Python module")
    return sha256_bytes(_normalized_python_source(module))


def canonical_sha256(value: Any) -> str:
    """Hash a value's canonical JSON representation."""

    return sha256_bytes(canonical_json_bytes(value))


__all__ = [
    "callable_source_sha256",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "module_source_sha256",
    "sha256_bytes",
]
