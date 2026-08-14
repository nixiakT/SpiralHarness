"""Opaque finite compiler and exact reconstruction verifier for v3."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, ValidationError, model_validator

from spiral_harness.benchmark._harness_fault_cases import (
    RepairRuleId,
    is_candidate_rule,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    ImmutableModel,
    Sha256,
)
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.storage.artifact_store import ArtifactStore

HARNESS_FAULT_COMPILATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-compilation.v3+json"
)
HARNESS_FAULT_ACTION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-repair-action.v3+json"
)
COMPILER_VERSION = "spiral-harness.harness-fault-compiler:v3-one-family"
CONTROL_MATCH_KIND = "exact-utf8-byte-length-v1"
_SCAFFOLD_PREFIX = "CONTROLLED_MIDDLEWARE_RULE\nPAYLOAD_BEGIN\n"
_SCAFFOLD_SUFFIX = "\nPAYLOAD_END\nReturn only the task's requested JSON."
_PAYLOAD_WIDTH = 384
_RULE_PAYLOADS = {
    RepairRuleId.RULE_00: (
        "Route every request through branch-1. Preserve exact LEFT and RIGHT bytes for "
        "comparison and observable construction."
    ),
    RepairRuleId.RULE_01: (
        "Route every request through branch-0. Strip outer Unicode whitespace and casefold "
        "LEFT and RIGHT before comparison and observable construction."
    ),
    RepairRuleId.RULE_02: (
        "Route canonical-policy requests through branch-0 and literal-policy requests through "
        "branch-1. Apply only the selected branch's transformation."
    ),
    RepairRuleId.CONTROL_00: (
        "Neutral control marker. Preserve the legacy branch-1 route for every request and add "
        "no canonicalization behavior."
    ),
}
_PARENT_PROMPT = (
    "LEGACY_MIDDLEWARE_RULE\nAlways route comparison requests through branch-1. Preserve "
    "LEFT and RIGHT exactly and emit only the requested JSON object."
)


class HarnessFaultCompilationError(ValueError):
    """A proposal or persisted compiler graph failed exact reconstruction."""


class HarnessRole(StrEnum):
    ORACLE = "oracle"
    FAULTY_PARENT = "faulty-parent"
    CANDIDATE = "candidate"
    REVERT = "revert"
    PLACEBO = "placebo"


class FaultRepairAction(ImmutableModel):
    """Entire model-authored surface: one opaque allowlisted ID."""

    schema_version: Literal["3"] = "3"
    rule_id: RepairRuleId

    @model_validator(mode="after")
    def only_candidate_catalog(self) -> Self:
        if not is_candidate_rule(self.rule_id):
            raise ValueError("rule_id is not in the candidate catalog")
        return self


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise HarnessFaultCompilationError(f"duplicate proposal key: {key!r}")
        value[key] = item
    return value


def parse_fault_repair_action(text: str) -> FaultRepairAction:
    """Accept exactly ``{"rule_id": <opaque-id>}`` and no prompt text."""

    if type(text) is not str:
        raise HarnessFaultCompilationError("repair proposal must be a string")
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, HarnessFaultCompilationError) as exc:
        raise HarnessFaultCompilationError(f"invalid repair proposal JSON: {exc}") from exc
    if type(value) is not dict or set(value) != {"rule_id"}:
        raise HarnessFaultCompilationError("proposal must contain only one opaque rule_id")
    if type(value["rule_id"]) is not str:
        raise HarnessFaultCompilationError("rule_id must be an exact string")
    try:
        return FaultRepairAction(rule_id=RepairRuleId(value["rule_id"]))
    except (ValueError, ValidationError) as exc:
        raise HarnessFaultCompilationError(f"invalid opaque rule selection: {exc}") from exc


def _scaffold_prompt(rule_id: RepairRuleId) -> str:
    payload = _RULE_PAYLOADS[rule_id]
    encoded = payload.encode("ascii")
    if len(encoded) > _PAYLOAD_WIDTH:  # pragma: no cover - frozen constants
        raise RuntimeError("frozen middleware payload exceeds scaffold width")
    padded = payload + " " * (_PAYLOAD_WIDTH - len(encoded))
    return _SCAFFOLD_PREFIX + padded + _SCAFFOLD_SUFFIX


COMPILER_FINGERPRINT = canonical_sha256(
    {
        "version": COMPILER_VERSION,
        "parent_prompt": _PARENT_PROMPT,
        "scaffold_prefix": _SCAFFOLD_PREFIX,
        "scaffold_suffix": _SCAFFOLD_SUFFIX,
        "payload_width": _PAYLOAD_WIDTH,
        "rule_payloads": tuple((rule.value, _RULE_PAYLOADS[rule]) for rule in RepairRuleId),
        "control_match": CONTROL_MATCH_KIND,
    }
)


class CompiledHarnessEntry(ImmutableModel):
    role: HarnessRole
    rule_id: RepairRuleId
    harness_ref: ArtifactRef
    prompt_ref: ArtifactRef

    @model_validator(mode="after")
    def exact_media(self) -> Self:
        if self.harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError("harness_ref declares the wrong media type")
        if self.prompt_ref.media_type != "text/plain":
            raise ValueError("prompt_ref must declare exact text/plain")
        return self


class HarnessFaultCompilationManifest(ImmutableModel):
    """Complete treatment/control graph emitted by the trusted compiler."""

    schema_version: Literal["3"] = "3"
    compiler_version: Literal["spiral-harness.harness-fault-compiler:v3-one-family"] = (
        COMPILER_VERSION
    )
    compiler_fingerprint: Literal[COMPILER_FINGERPRINT] = COMPILER_FINGERPRINT
    model_spec_fingerprint: Sha256
    runtime_producer_id: Sha256
    action_ref: ArtifactRef
    selected_rule_id: RepairRuleId
    control_match_kind: Literal["exact-utf8-byte-length-v1"] = CONTROL_MATCH_KIND
    matched_prompt_bytes: Annotated[int, Field(ge=1, strict=True)]
    entries: Annotated[tuple[CompiledHarnessEntry, ...], Field(min_length=5, max_length=5)]

    @model_validator(mode="after")
    def complete_roles(self) -> Self:
        if self.action_ref.media_type != HARNESS_FAULT_ACTION_MEDIA_TYPE:
            raise ValueError("action_ref declares the wrong media type")
        if len(self.entries) != 5 or {entry.role for entry in self.entries} != set(HarnessRole):
            raise ValueError("compilation requires exactly five roles")
        if len({entry.harness_ref.sha256 for entry in self.entries}) != 5:
            raise ValueError("each role requires a distinct harness manifest")
        if self.entry(HarnessRole.CANDIDATE).rule_id is not self.selected_rule_id:
            raise ValueError("candidate rule differs from selected_rule_id")
        return self

    def entry(self, role: HarnessRole) -> CompiledHarnessEntry:
        for entry in self.entries:
            if entry.role is role:
                return entry
        raise KeyError(role)

    def role_for_harness(self, ref: ArtifactRef) -> HarnessRole:
        matches = tuple(item.role for item in self.entries if item.harness_ref == ref)
        if len(matches) != 1:
            raise KeyError("harness is outside the compilation graph")
        return matches[0]


class HarnessFaultCompilationRecord(ImmutableModel):
    manifest: HarnessFaultCompilationManifest
    manifest_ref: ArtifactRef

    @model_validator(mode="after")
    def exact_media(self) -> Self:
        if self.manifest_ref.media_type != HARNESS_FAULT_COMPILATION_MEDIA_TYPE:
            raise ValueError("manifest_ref declares the wrong media type")
        return self


def _bytes_ref(payload: bytes, media_type: str) -> ArtifactRef:
    return ArtifactRef(sha256=sha256_bytes(payload), size=len(payload), media_type=media_type)


def _json_ref(value: object, media_type: str) -> ArtifactRef:
    return _bytes_ref(canonical_json_bytes(value), media_type)


def _manifest(
    spec: FrozenModelSpec,
    prompt_ref: ArtifactRef,
    parent_ref: ArtifactRef | None,
) -> HarnessManifest:
    return HarnessManifest(
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        trusted_plane_version=COMPILER_VERSION,
        parent=parent_ref,
        components=(
            HarnessComponentRef(
                name="system-prompt",
                kind=ComponentKind.PROMPT,
                artifact=prompt_ref,
            ),
        ),
    )


def _expected_compilation(
    spec: FrozenModelSpec,
    action: FaultRepairAction,
    action_ref: ArtifactRef,
    runtime_producer_id: str,
) -> tuple[
    HarnessFaultCompilationManifest,
    dict[ArtifactRef, bytes],
    dict[ArtifactRef, HarnessManifest],
]:
    parent_bytes = _PARENT_PROMPT.encode("utf-8")
    oracle_bytes = _scaffold_prompt(RepairRuleId.RULE_02).encode("utf-8")
    candidate_bytes = _scaffold_prompt(action.rule_id).encode("utf-8")
    placebo_bytes = _scaffold_prompt(RepairRuleId.CONTROL_00).encode("utf-8")
    if len(candidate_bytes) != len(placebo_bytes):  # pragma: no cover - frozen scaffold
        raise RuntimeError("candidate/placebo scaffold byte lengths differ")
    parent_prompt_ref = _bytes_ref(parent_bytes, "text/plain")
    oracle_prompt_ref = _bytes_ref(oracle_bytes, "text/plain")
    candidate_prompt_ref = _bytes_ref(candidate_bytes, "text/plain")
    placebo_prompt_ref = _bytes_ref(placebo_bytes, "text/plain")

    parent_manifest = _manifest(spec, parent_prompt_ref, None)
    parent_ref = _json_ref(parent_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    # The oracle is a fixed standalone upper-bound control, not a mutation selected by
    # the search policy. Keeping it root-scoped preserves role provenance even when
    # the candidate selects the oracle's exact rule/prompt bytes.
    oracle_manifest = _manifest(spec, oracle_prompt_ref, None)
    oracle_ref = _json_ref(oracle_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    candidate_manifest = _manifest(spec, candidate_prompt_ref, parent_ref)
    candidate_ref = _json_ref(candidate_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    # Revert uses the exact parent prompt artifact bytes and a candidate parent edge.
    revert_manifest = _manifest(spec, parent_prompt_ref, candidate_ref)
    revert_ref = _json_ref(revert_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    placebo_manifest = _manifest(spec, placebo_prompt_ref, parent_ref)
    placebo_ref = _json_ref(placebo_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    entries = (
        CompiledHarnessEntry(
            role=HarnessRole.ORACLE,
            rule_id=RepairRuleId.RULE_02,
            harness_ref=oracle_ref,
            prompt_ref=oracle_prompt_ref,
        ),
        CompiledHarnessEntry(
            role=HarnessRole.FAULTY_PARENT,
            rule_id=RepairRuleId.RULE_00,
            harness_ref=parent_ref,
            prompt_ref=parent_prompt_ref,
        ),
        CompiledHarnessEntry(
            role=HarnessRole.CANDIDATE,
            rule_id=action.rule_id,
            harness_ref=candidate_ref,
            prompt_ref=candidate_prompt_ref,
        ),
        CompiledHarnessEntry(
            role=HarnessRole.REVERT,
            rule_id=RepairRuleId.RULE_00,
            harness_ref=revert_ref,
            prompt_ref=parent_prompt_ref,
        ),
        CompiledHarnessEntry(
            role=HarnessRole.PLACEBO,
            rule_id=RepairRuleId.CONTROL_00,
            harness_ref=placebo_ref,
            prompt_ref=placebo_prompt_ref,
        ),
    )
    compilation = HarnessFaultCompilationManifest(
        model_spec_fingerprint=spec.fingerprint,
        runtime_producer_id=runtime_producer_id,
        action_ref=action_ref,
        selected_rule_id=action.rule_id,
        matched_prompt_bytes=len(candidate_bytes),
        entries=tuple(sorted(entries, key=lambda item: item.role.value)),
    )
    prompts = {
        parent_prompt_ref: parent_bytes,
        oracle_prompt_ref: oracle_bytes,
        candidate_prompt_ref: candidate_bytes,
        placebo_prompt_ref: placebo_bytes,
    }
    manifests = {
        parent_ref: parent_manifest,
        oracle_ref: oracle_manifest,
        candidate_ref: candidate_manifest,
        revert_ref: revert_manifest,
        placebo_ref: placebo_manifest,
    }
    return compilation, prompts, manifests


def compile_fault_repair(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    action: FaultRepairAction,
    *,
    runtime_producer_id: str,
) -> HarnessFaultCompilationRecord:
    """Publish only the deterministic graph reconstructed from one opaque action."""

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    checked_action = FaultRepairAction.model_validate(action, strict=True)
    action_ref = store.put_json(checked_action, media_type=HARNESS_FAULT_ACTION_MEDIA_TYPE)
    compilation, prompts, manifests = _expected_compilation(
        checked_spec, checked_action, action_ref, runtime_producer_id
    )
    for expected_ref, payload in prompts.items():
        if store.put_bytes(payload, media_type="text/plain") != expected_ref:
            raise HarnessFaultCompilationError("prompt publication changed its exact bytes")
    for expected_ref, manifest in manifests.items():
        if store.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE) != expected_ref:
            raise HarnessFaultCompilationError("harness publication changed its graph")
    manifest_ref = store.put_json(compilation, media_type=HARNESS_FAULT_COMPILATION_MEDIA_TYPE)
    return HarnessFaultCompilationRecord(manifest=compilation, manifest_ref=manifest_ref)


def verify_fault_compilation(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    compilation_ref: ArtifactRef,
    *,
    expected_runtime_producer_id: str,
) -> HarnessFaultCompilationManifest:
    """Rebuild exact action/prompts/manifests/parent graph; reject self-consistent forgeries."""

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    try:
        if compilation_ref.media_type != HARNESS_FAULT_COMPILATION_MEDIA_TYPE:
            raise ValueError("wrong compilation media type")
        actual = store.get_json(compilation_ref, HarnessFaultCompilationManifest)
        action = store.get_json(actual.action_ref, FaultRepairAction)
    except Exception as exc:
        raise HarnessFaultCompilationError("compilation or action cannot be verified") from exc
    expected, prompts, manifests = _expected_compilation(
        checked_spec,
        action,
        actual.action_ref,
        expected_runtime_producer_id,
    )
    if actual != expected or compilation_ref != _json_ref(
        expected, HARNESS_FAULT_COMPILATION_MEDIA_TYPE
    ):
        raise HarnessFaultCompilationError("compilation differs from frozen reconstruction")
    try:
        for ref, payload in prompts.items():
            if store.get_bytes(ref) != payload:
                raise ValueError("prompt bytes differ")
        for ref, manifest in manifests.items():
            if store.get_json(ref, HarnessManifest) != manifest:
                raise ValueError("manifest graph differs")
    except Exception as exc:
        raise HarnessFaultCompilationError(
            "compiled prompt/manifest graph cannot be replayed"
        ) from exc
    parent = actual.entry(HarnessRole.FAULTY_PARENT)
    candidate = actual.entry(HarnessRole.CANDIDATE)
    revert = actual.entry(HarnessRole.REVERT)
    placebo = actual.entry(HarnessRole.PLACEBO)
    if revert.prompt_ref != parent.prompt_ref:
        raise HarnessFaultCompilationError("revert does not reuse exact parent prompt bytes")
    if candidate.prompt_ref.size != placebo.prompt_ref.size:
        raise HarnessFaultCompilationError("placebo is not exact UTF-8-byte-length matched")
    return actual


__all__ = [
    "COMPILER_FINGERPRINT",
    "COMPILER_VERSION",
    "CONTROL_MATCH_KIND",
    "HARNESS_FAULT_ACTION_MEDIA_TYPE",
    "HARNESS_FAULT_COMPILATION_MEDIA_TYPE",
    "CompiledHarnessEntry",
    "FaultRepairAction",
    "HarnessFaultCompilationError",
    "HarnessFaultCompilationManifest",
    "HarnessFaultCompilationRecord",
    "HarnessRole",
    "compile_fault_repair",
    "parse_fault_repair_action",
    "verify_fault_compilation",
]
