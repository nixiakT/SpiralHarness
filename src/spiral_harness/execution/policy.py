"""Deny-by-default capability contracts for untrusted harness workers."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.models import ImmutableModel, NonEmptyStr

CAPABILITY_POLICY_MEDIA_TYPE = "application/vnd.spiral-harness.capability-policy.v1+json"


class CapabilityKind(StrEnum):
    """Side-effecting authorities that a worker may request."""

    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    NETWORK = "network"
    PERSISTENT_STATE = "persistent_state"
    SECRET = "secret"
    SUBPROCESS = "subprocess"
    TOOL = "tool"


class CapabilityGrant(ImmutableModel):
    """An exact resource allowlist for one capability kind.

    Resource identifiers are opaque, adapter-owned names. They are matched by
    exact equality; globbing, prefix matching, and an implicit wildcard are
    deliberately unsupported at this trust boundary.
    """

    kind: CapabilityKind
    resources: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    max_uses: Annotated[int, Field(ge=1, strict=True)] | None = None

    @field_validator("resources")
    @classmethod
    def canonicalize_resources(cls, resources: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(resources))
        if len(ordered) != len(set(ordered)):
            raise ValueError("capability resources must be unique")
        return ordered


class CapabilityPolicy(ImmutableModel):
    """Frozen capability manifest interpreted by a trusted worker launcher."""

    schema_version: Literal["1"] = "1"
    default: Literal["deny"] = "deny"
    grants: tuple[CapabilityGrant, ...] = ()

    @field_validator("grants")
    @classmethod
    def canonicalize_grants(
        cls,
        grants: tuple[CapabilityGrant, ...],
    ) -> tuple[CapabilityGrant, ...]:
        return tuple(sorted(grants, key=lambda grant: grant.kind.value))

    @model_validator(mode="after")
    def one_grant_per_capability(self) -> CapabilityPolicy:
        kinds = tuple(grant.kind for grant in self.grants)
        if len(kinds) != len(set(kinds)):
            raise ValueError("a capability policy may contain only one grant per kind")
        return self

    def allows(self, kind: CapabilityKind, resource: str) -> bool:
        """Return whether *resource* is explicitly granted for *kind*."""

        if not isinstance(kind, CapabilityKind):
            raise TypeError("kind must be a CapabilityKind")
        if not isinstance(resource, str):
            raise TypeError("resource must be a string")
        for grant in self.grants:
            if grant.kind is kind:
                return resource in grant.resources
        return False


__all__ = [
    "CAPABILITY_POLICY_MEDIA_TYPE",
    "CapabilityGrant",
    "CapabilityKind",
    "CapabilityPolicy",
]
