"""Opaque multi-surface compiler and reconstruction verifier for v4."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from spiral_harness.benchmark._harness_fault_cases import RepairRuleId, is_candidate_rule
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.storage.artifact_store import ArtifactStore

HARNESS_FAULT_COMPILATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-compilation.v4+json"
)
HARNESS_FAULT_ACTION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-repair-action.v4+json"
)
COMPILER_VERSION = "spiral-harness.harness-fault-compiler:v4-multi-surface"
CONTROL_MATCH_KIND = "per-component-exact-utf8-byte-length-v2"
_SCAFFOLD_PREFIX = "HFB4_FROZEN_COMPONENT\nPAYLOAD_BEGIN\n"
_SCAFFOLD_SUFFIX = "\nPAYLOAD_END\n"
_PAYLOAD_WIDTH = 2_048
_COMPONENT_SPECS = (
    ("system-prompt", ComponentKind.PROMPT),
    ("retrieval-policy", ComponentKind.MEMORY),
    ("tool-schema", ComponentKind.TOOL),
    ("tool-router", ComponentKind.TOOL),
    ("middleware-policy", ComponentKind.MIDDLEWARE),
    ("control-flow-policy", ComponentKind.CONTROL_FLOW),
    ("skill-scope", ComponentKind.SKILL),
)
_RULE_PAYLOADS = {
    RepairRuleId.CONSTANT_LEGACY: (
        "Use the legacy branch for every family and context; no scoped repair is active."
    ),
    RepairRuleId.CONSTANT_SAFE: (
        "Use the safe branch for every family and context; ignore null and shift controls."
    ),
    RepairRuleId.ROUTED_POLICY: (
        "Route each frozen family/context through its registered safe or legacy branch."
    ),
    RepairRuleId.PROMPT_MEMORY_PATCH: (
        "Repair prompt precedence and memory isolation only; retain legacy behavior elsewhere."
    ),
    RepairRuleId.TOOL_PATCH: (
        "Repair tool schema and tool routing only; retain legacy behavior elsewhere."
    ),
    RepairRuleId.RUNTIME_PATCH: (
        "Repair middleware, control-flow, and skill scope only; retain legacy elsewhere."
    ),
    RepairRuleId.CONTROL_NEUTRAL: (
        "Length-matched neutral control; preserve the legacy branch for every request."
    ),
}

_MODEL_SOLVER_PROTOCOL = """
Return exactly one compact JSON object with keys `answer` and `observable`; no
markdown or extra keys. Read FAULT_FAMILY, CONTEXT, PRIMARY, and SECONDARY from
the user message. The RULE line selects a branch. CONSTANT_LEGACY and
CONTROL_NEUTRAL always select LEGACY. CONSTANT_SAFE always selects SAFE.
ROUTED_POLICY selects LEGACY in context-1, SAFE in context-0, and in context-x
selects SAFE only for prompt-instruction-precedence, tool-schema-validation,
middleware-canonicalization, and skill-scope-activation. A family patch selects
LEGACY in context-1/context-x; in context-0 it selects SAFE only for its named
families (PROMPT_MEMORY: prompt+memory, TOOL: both tool families, RUNTIME:
middleware+control-flow+skill), otherwise LEGACY. SAFE answers MATCH for
prompt-instruction-precedence, tool-schema-validation,
middleware-canonicalization, skill-scope-activation and DIFFERENT otherwise;
LEGACY reverses that. SAFE observable labels are trusted-instruction,
scoped-memory, validated-schema, authorized-route, guarded-flow, scoped-skill
and use PRIMARY. LEGACY labels are payload-instruction, global-memory,
unchecked-schema, fallback-route, legacy-flow, default-skill and use SECONDARY.
Format those observables as label=<compact JSON string>. For
middleware-canonicalization the sole label is compared: SAFE compares
strip+casefold(PRIMARY)|strip+casefold(SECONDARY), LEGACY compares the exact raw
PRIMARY|SECONDARY, again encoded as one compact JSON string.
""".strip()


class HarnessFaultCompilationError(ValueError):
    """A proposal or persisted compiler graph failed exact reconstruction."""


class HarnessRole(StrEnum):
    ORACLE = "oracle"
    FAULTY_PARENT = "faulty-parent"
    CANDIDATE = "candidate"
    REVERT = "revert"
    PLACEBO = "placebo"


class FaultRepairAction(ImmutableModel):
    """Entire model-authored surface: one opaque allowlisted rule ID."""

    schema_version: Literal["4"] = "4"
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
    """Accept exactly ``{"rule_id": <opaque-id>}`` and no free-form patch."""

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


def _component_payload(name: str, rule_id: RepairRuleId) -> bytes:
    solver_protocol = (
        f"\nMODEL_SOLVER_PROTOCOL:\n{_MODEL_SOLVER_PROTOCOL}" if name == "system-prompt" else ""
    )
    payload = f"COMPONENT={name}\nRULE={rule_id.value}\n{_RULE_PAYLOADS[rule_id]}{solver_protocol}"
    encoded = payload.encode("ascii")
    if len(encoded) > _PAYLOAD_WIDTH:  # pragma: no cover - frozen constants
        raise RuntimeError("frozen component payload exceeds scaffold width")
    padded = payload + " " * (_PAYLOAD_WIDTH - len(encoded))
    return (_SCAFFOLD_PREFIX + padded + _SCAFFOLD_SUFFIX).encode("utf-8")


COMPILER_FINGERPRINT = canonical_sha256(
    {
        "version": COMPILER_VERSION,
        "component_specs": tuple((name, kind.value) for name, kind in _COMPONENT_SPECS),
        "scaffold_prefix": _SCAFFOLD_PREFIX,
        "scaffold_suffix": _SCAFFOLD_SUFFIX,
        "payload_width": _PAYLOAD_WIDTH,
        "rule_payloads": tuple((rule.value, _RULE_PAYLOADS[rule]) for rule in RepairRuleId),
        "model_solver_protocol": _MODEL_SOLVER_PROTOCOL,
        "control_match": CONTROL_MATCH_KIND,
    }
)


class MatchedComponentBytes(ImmutableModel):
    name: NonEmptyStr
    kind: ComponentKind
    byte_count: Annotated[int, Field(ge=1, strict=True)]


class CompiledHarnessEntry(ImmutableModel):
    role: HarnessRole
    rule_id: RepairRuleId
    harness_ref: ArtifactRef
    surface_components: Annotated[
        tuple[HarnessComponentRef, ...],
        Field(min_length=7, max_length=7),
    ]

    @field_validator("surface_components")
    @classmethod
    def canonical_components(
        cls, values: tuple[HarnessComponentRef, ...]
    ) -> tuple[HarnessComponentRef, ...]:
        return tuple(sorted(values, key=lambda item: item.name))

    @model_validator(mode="after")
    def exact_media_and_surface_set(self) -> Self:
        if self.harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError("harness_ref declares the wrong media type")
        expected = {(name, kind) for name, kind in _COMPONENT_SPECS}
        actual = {(item.name, item.kind) for item in self.surface_components}
        if len(self.surface_components) != len(expected) or actual != expected:
            raise ValueError("entry does not contain the exact multi-surface component set")
        if any(item.artifact.media_type != "text/plain" for item in self.surface_components):
            raise ValueError("surface component must declare exact text/plain")
        return self

    def component(self, name: str) -> HarnessComponentRef:
        for item in self.surface_components:
            if item.name == name:
                return item
        raise KeyError(name)

    @property
    def prompt_ref(self) -> ArtifactRef:
        return self.component("system-prompt").artifact


class HarnessFaultCompilationManifest(ImmutableModel):
    schema_version: Literal["4"] = "4"
    compiler_version: Literal["spiral-harness.harness-fault-compiler:v4-multi-surface"] = (
        COMPILER_VERSION
    )
    compiler_fingerprint: Literal[COMPILER_FINGERPRINT] = COMPILER_FINGERPRINT
    model_spec_fingerprint: Sha256
    runtime_producer_id: Sha256
    action_ref: ArtifactRef
    selected_rule_id: RepairRuleId
    control_match_kind: Literal["per-component-exact-utf8-byte-length-v2"] = CONTROL_MATCH_KIND
    matched_component_bytes: Annotated[
        tuple[MatchedComponentBytes, ...],
        Field(min_length=7, max_length=7),
    ]
    entries: Annotated[tuple[CompiledHarnessEntry, ...], Field(min_length=5, max_length=5)]

    @model_validator(mode="after")
    def complete_roles_and_components(self) -> Self:
        if self.action_ref.media_type != HARNESS_FAULT_ACTION_MEDIA_TYPE:
            raise ValueError("action_ref declares the wrong media type")
        if len(self.entries) != 5 or {entry.role for entry in self.entries} != set(HarnessRole):
            raise ValueError("compilation requires exactly five roles")
        if len({entry.harness_ref.sha256 for entry in self.entries}) != 5:
            raise ValueError("each role requires a distinct harness manifest")
        if self.entry(HarnessRole.CANDIDATE).rule_id is not self.selected_rule_id:
            raise ValueError("candidate rule differs from selected_rule_id")
        expected = {(name, kind) for name, kind in _COMPONENT_SPECS}
        actual = {(item.name, item.kind) for item in self.matched_component_bytes}
        if actual != expected or len(self.matched_component_bytes) != len(expected):
            raise ValueError("matched bytes omit a mutable surface component")
        return self

    @property
    def matched_prompt_bytes(self) -> int:
        for item in self.matched_component_bytes:
            if item.name == "system-prompt":
                return item.byte_count
        raise KeyError("system-prompt")

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


def _surface_components(rule_id: RepairRuleId) -> tuple[HarnessComponentRef, ...]:
    return tuple(
        HarnessComponentRef(
            name=name,
            kind=kind,
            artifact=_bytes_ref(_component_payload(name, rule_id), "text/plain"),
        )
        for name, kind in _COMPONENT_SPECS
    )


def _execution_manifest(
    spec: FrozenModelSpec,
    components: tuple[HarnessComponentRef, ...],
    parent_ref: ArtifactRef | None,
) -> HarnessManifest:
    """Project the benchmark graph to the one prompt the generic runner can inject.

    The complete seven-component graph remains digest-bound in the compilation
    manifest and is consumed by the trusted benchmark runtime.  The generic
    runner intentionally supports only prompt/skill materialization.
    """

    prompt = next(item for item in components if item.name == "system-prompt")
    return HarnessManifest(
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        trusted_plane_version=COMPILER_VERSION,
        parent=parent_ref,
        components=(prompt,),
    )


def _entry(
    role: HarnessRole,
    rule_id: RepairRuleId,
    harness_ref: ArtifactRef,
    components: tuple[HarnessComponentRef, ...],
) -> CompiledHarnessEntry:
    return CompiledHarnessEntry(
        role=role,
        rule_id=rule_id,
        harness_ref=harness_ref,
        surface_components=components,
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
    component_sets = {
        HarnessRole.FAULTY_PARENT: _surface_components(RepairRuleId.CONSTANT_LEGACY),
        HarnessRole.ORACLE: _surface_components(RepairRuleId.ROUTED_POLICY),
        HarnessRole.CANDIDATE: _surface_components(action.rule_id),
        HarnessRole.PLACEBO: _surface_components(RepairRuleId.CONTROL_NEUTRAL),
    }
    component_sets[HarnessRole.REVERT] = component_sets[HarnessRole.FAULTY_PARENT]
    parent_manifest = _execution_manifest(spec, component_sets[HarnessRole.FAULTY_PARENT], None)
    parent_ref = _json_ref(parent_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    oracle_manifest = _execution_manifest(spec, component_sets[HarnessRole.ORACLE], None)
    oracle_ref = _json_ref(oracle_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    candidate_manifest = _execution_manifest(
        spec, component_sets[HarnessRole.CANDIDATE], parent_ref
    )
    candidate_ref = _json_ref(candidate_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    revert_manifest = _execution_manifest(spec, component_sets[HarnessRole.REVERT], candidate_ref)
    revert_ref = _json_ref(revert_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    placebo_manifest = _execution_manifest(spec, component_sets[HarnessRole.PLACEBO], parent_ref)
    placebo_ref = _json_ref(placebo_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    role_values = {
        HarnessRole.ORACLE: (RepairRuleId.ROUTED_POLICY, oracle_ref),
        HarnessRole.FAULTY_PARENT: (RepairRuleId.CONSTANT_LEGACY, parent_ref),
        HarnessRole.CANDIDATE: (action.rule_id, candidate_ref),
        HarnessRole.REVERT: (RepairRuleId.CONSTANT_LEGACY, revert_ref),
        HarnessRole.PLACEBO: (RepairRuleId.CONTROL_NEUTRAL, placebo_ref),
    }
    entries = tuple(
        _entry(role, rule, ref, component_sets[role]) for role, (rule, ref) in role_values.items()
    )
    candidate_components = component_sets[HarnessRole.CANDIDATE]
    placebo_components = component_sets[HarnessRole.PLACEBO]
    if tuple(item.artifact.size for item in candidate_components) != tuple(
        item.artifact.size for item in placebo_components
    ):  # pragma: no cover - frozen scaffold
        raise RuntimeError("candidate/placebo component byte lengths differ")
    matched = tuple(
        MatchedComponentBytes(
            name=item.name,
            kind=item.kind,
            byte_count=item.artifact.size,
        )
        for item in candidate_components
    )
    compilation = HarnessFaultCompilationManifest(
        model_spec_fingerprint=spec.fingerprint,
        runtime_producer_id=runtime_producer_id,
        action_ref=action_ref,
        selected_rule_id=action.rule_id,
        matched_component_bytes=matched,
        entries=tuple(sorted(entries, key=lambda item: item.role.value)),
    )
    payloads = {
        component.artifact: _component_payload(component.name, rule)
        for role, (rule, _) in role_values.items()
        for component in component_sets[role]
    }
    manifests = {
        parent_ref: parent_manifest,
        oracle_ref: oracle_manifest,
        candidate_ref: candidate_manifest,
        revert_ref: revert_manifest,
        placebo_ref: placebo_manifest,
    }
    return compilation, payloads, manifests


def compile_fault_repair(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    action: FaultRepairAction,
    *,
    runtime_producer_id: str,
) -> HarnessFaultCompilationRecord:
    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    checked_action = FaultRepairAction.model_validate(action, strict=True)
    action_ref = store.put_json(checked_action, media_type=HARNESS_FAULT_ACTION_MEDIA_TYPE)
    compilation, payloads, manifests = _expected_compilation(
        checked_spec, checked_action, action_ref, runtime_producer_id
    )
    for expected_ref, payload in payloads.items():
        if store.put_bytes(payload, media_type="text/plain") != expected_ref:
            raise HarnessFaultCompilationError("component publication changed exact bytes")
    for expected_ref, manifest in manifests.items():
        if store.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE) != expected_ref:
            raise HarnessFaultCompilationError("harness publication changed its graph")
    manifest_ref = store.put_json(compilation, media_type=HARNESS_FAULT_COMPILATION_MEDIA_TYPE)
    return HarnessFaultCompilationRecord(manifest=compilation, manifest_ref=manifest_ref)


def publish_faulty_parent_harness(
    store: ArtifactStore,
    spec: FrozenModelSpec,
) -> ArtifactRef:
    """Publish the candidate-independent faulty parent before proposal generation.

    The returned manifest is byte-identical to the parent reconstructed by any
    later :func:`compile_fault_repair` call for the same frozen model spec.
    """

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    components = _surface_components(RepairRuleId.CONSTANT_LEGACY)
    manifest = _execution_manifest(checked_spec, components, None)
    prompt = next(item for item in components if item.name == "system-prompt")
    expected_prompt = _component_payload(prompt.name, RepairRuleId.CONSTANT_LEGACY)
    if store.put_bytes(expected_prompt, media_type="text/plain") != prompt.artifact:
        raise HarnessFaultCompilationError("parent prompt publication changed exact bytes")
    ref = store.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    expected_ref = _json_ref(manifest, HARNESS_MANIFEST_MEDIA_TYPE)
    if ref != expected_ref:
        raise HarnessFaultCompilationError("parent harness publication changed its graph")
    return ref


def verify_fault_compilation(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    compilation_ref: ArtifactRef,
    *,
    expected_runtime_producer_id: str,
) -> HarnessFaultCompilationManifest:
    """Rebuild every action, component, manifest, and parent edge."""

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
    expected, payloads, manifests = _expected_compilation(
        checked_spec, action, actual.action_ref, expected_runtime_producer_id
    )
    if actual != expected or compilation_ref != _json_ref(
        expected, HARNESS_FAULT_COMPILATION_MEDIA_TYPE
    ):
        raise HarnessFaultCompilationError("compilation differs from frozen reconstruction")
    try:
        for ref, payload in payloads.items():
            if store.get_bytes(ref) != payload:
                raise ValueError("component bytes differ")
        for ref, manifest in manifests.items():
            if store.get_json(ref, HarnessManifest) != manifest:
                raise ValueError("manifest graph differs")
    except Exception as exc:
        raise HarnessFaultCompilationError(
            "compiled component/manifest graph cannot be replayed"
        ) from exc
    parent = actual.entry(HarnessRole.FAULTY_PARENT)
    candidate = actual.entry(HarnessRole.CANDIDATE)
    revert = actual.entry(HarnessRole.REVERT)
    placebo = actual.entry(HarnessRole.PLACEBO)
    if revert.surface_components != parent.surface_components:
        raise HarnessFaultCompilationError("revert does not reuse exact parent components")
    if tuple(item.artifact.size for item in candidate.surface_components) != tuple(
        item.artifact.size for item in placebo.surface_components
    ):
        raise HarnessFaultCompilationError("placebo is not per-component byte-length matched")
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
    "MatchedComponentBytes",
    "compile_fault_repair",
    "parse_fault_repair_action",
    "publish_faulty_parent_harness",
    "verify_fault_compilation",
]
