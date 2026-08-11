from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.execution.policy import (
    CapabilityGrant,
    CapabilityKind,
    CapabilityPolicy,
)


def test_capability_policy_denies_everything_by_default() -> None:
    policy = CapabilityPolicy()

    for kind in CapabilityKind:
        assert not policy.allows(kind, "any-resource")


def test_capability_policy_uses_exact_sorted_allowlists() -> None:
    policy = CapabilityPolicy(
        grants=(
            CapabilityGrant(
                kind=CapabilityKind.TOOL,
                resources=("search", "calculator"),
                max_uses=3,
            ),
            CapabilityGrant(
                kind=CapabilityKind.FILESYSTEM_READ,
                resources=("task-input",),
            ),
        )
    )

    assert tuple(grant.kind for grant in policy.grants) == (
        CapabilityKind.FILESYSTEM_READ,
        CapabilityKind.TOOL,
    )
    assert policy.grants[1].resources == ("calculator", "search")
    assert policy.allows(CapabilityKind.TOOL, "search")
    assert not policy.allows(CapabilityKind.TOOL, "search-index")
    assert not policy.allows(CapabilityKind.NETWORK, "search")


def test_capability_policy_rejects_duplicate_or_empty_grants() -> None:
    with pytest.raises(ValidationError, match="resources must be unique"):
        CapabilityGrant(
            kind=CapabilityKind.TOOL,
            resources=("search", "search"),
        )
    with pytest.raises(ValidationError):
        CapabilityGrant(kind=CapabilityKind.TOOL, resources=())
    with pytest.raises(ValidationError, match="one grant per kind"):
        CapabilityPolicy(
            grants=(
                CapabilityGrant(kind=CapabilityKind.TOOL, resources=("a",)),
                CapabilityGrant(kind=CapabilityKind.TOOL, resources=("b",)),
            )
        )


def test_capability_policy_is_strict_canonical_and_has_no_magic_wildcard() -> None:
    with pytest.raises(ValidationError):
        CapabilityGrant(kind="tool", resources=("search",))
    with pytest.raises(TypeError, match="CapabilityKind"):
        CapabilityPolicy().allows("tool", "search")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="resource"):
        CapabilityPolicy().allows(CapabilityKind.TOOL, 1)  # type: ignore[arg-type]

    wildcard = CapabilityPolicy(
        grants=(CapabilityGrant(kind=CapabilityKind.NETWORK, resources=("*",)),)
    )
    assert wildcard.allows(CapabilityKind.NETWORK, "*")
    assert not wildcard.allows(CapabilityKind.NETWORK, "example.com")
    assert CapabilityPolicy.model_validate_json(canonical_json_bytes(wildcard)) == wildcard
