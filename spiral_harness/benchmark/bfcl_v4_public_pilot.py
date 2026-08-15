"""Source-isolated loading and provider-free mechanics for the BFCL V4 pilot.

Nothing in this module reads a possible-answer file or invokes a model.  The
pilot is public/development-only because BFCL itself publishes those answers;
the isolation here is an experimental information-flow control, not secrecy.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any

from spiral_harness.benchmark.bfcl_v4_fixture_bridge import (
    BfclV4FixtureBridgeError,
    _checkout_and_git,
    _git_blob,
)
from spiral_harness.benchmark.bfcl_v4_fixture_contracts import BfclV4SourceFileBinding
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4AdapterSourceVerification,
    BfclV4LoadedPublicPilot,
    BfclV4OpenAiToolAdapterResult,
    BfclV4PublicPilotTask,
    BfclV4PureAtBSample,
    BfclV4PureAtBSelection,
    BfclV4ToolNameBinding,
    BfclV4WireToolCall,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_identities import (
    BFCL_V4_PILOT_ADAPTER_SOURCE_IDENTITIES,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256, sha256_bytes

BFCL_V4_GORILLA_TO_OPENAPI = MappingProxyType(
    {
        "integer": "integer",
        "number": "number",
        "float": "number",
        "string": "string",
        "boolean": "boolean",
        "bool": "boolean",
        "array": "array",
        "list": "array",
        "dict": "object",
        "object": "object",
        "tuple": "array",
        "any": "string",
        "byte": "integer",
        "short": "integer",
        "long": "integer",
        "double": "number",
        "char": "string",
        "ArrayList": "array",
        "Array": "array",
        "HashMap": "object",
        "Hashtable": "object",
        "Queue": "array",
        "Stack": "array",
        "Any": "string",
        "String": "string",
        "Bigint": "integer",
    }
)


class BfclV4PublicPilotError(RuntimeError):
    """A pinned public-pilot contract failed closed."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_surrogates(value: Any) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON strings must not contain surrogate code points")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_surrogates(key)
            _reject_surrogates(item)
    elif isinstance(value, list):
        for item in value:
            _reject_surrogates(item)


def _strict_json(text: str, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate object keys")
            output[key] = value
        return output

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    _reject_surrogates(value)
    canonical_json(value)  # also rejects finite-overflow floats such as 1e400
    return value


def _read_pinned_blob(git: Path, checkout: Path, git_path: str) -> bytes:
    """Read one exact-commit Git object; kept separate for no-answer auditing."""

    return _git_blob(git, checkout, git_path)


def _verified_blob(git: Path, checkout: Path, git_path: str, size: int, digest: str) -> bytes:
    if "possible_answer" in git_path:
        raise BfclV4PublicPilotError("candidate-facing pilot cannot read possible-answer paths")
    content = _read_pinned_blob(git, checkout, git_path)
    if len(content) != size or sha256_bytes(content) != digest:
        raise BfclV4PublicPilotError(f"pinned BFCL source differs: {git_path}")
    return content


def _selected_jsonl_rows(
    content: bytes,
    selected_ids: frozenset[str],
    label: str,
) -> dict[str, tuple[dict[str, Any], bytes]]:
    selected: dict[str, tuple[dict[str, Any], bytes]] = {}
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.endswith(b"\r\n") or (not line.endswith(b"\n") and index != len(lines) - 1):
            raise BfclV4PublicPilotError(f"{label} is not canonical LF-delimited JSONL")
        payload = line[:-1] if line.endswith(b"\n") else line
        try:
            value = _strict_json(payload.decode("utf-8"), label)
        except (UnicodeDecodeError, ValueError) as error:
            raise BfclV4PublicPilotError(f"{label} contains invalid JSONL") from error
        if isinstance(value, dict) and value.get("id") in selected_ids:
            task_id = value["id"]
            if task_id in selected:
                raise BfclV4PublicPilotError(f"{label} repeats selected task {task_id}")
            selected[task_id] = (value, line)
    if set(selected) != set(selected_ids):
        raise BfclV4PublicPilotError(f"{label} does not contain the exact frozen task subset")
    return selected


def load_bfcl_v4_public_pilot(checkout: str | Path) -> BfclV4LoadedPublicPilot:
    """Load only question/schema objects for the frozen public-development roster."""

    try:
        resolved, git = _checkout_and_git(checkout)
    except BfclV4FixtureBridgeError as error:
        raise BfclV4PublicPilotError("pinned BFCL checkout verification failed") from error
    roster = BFCL_V4_PUBLIC_PILOT_MANIFEST.roster
    rows: dict[str, tuple[dict[str, Any], bytes, int, str]] = {}
    for git_path in sorted({item.question_git_path for item in roster}):
        matching_entries = tuple(item for item in roster if item.question_git_path == git_path)
        source_identities = {
            (item.question_blob_size, item.question_blob_sha256) for item in matching_entries
        }
        if len(source_identities) != 1:
            raise BfclV4PublicPilotError("roster entries disagree on question blob identity")
        size, digest = next(iter(source_identities))
        content = _verified_blob(git, resolved, git_path, size, digest)
        selected_ids = frozenset(item.task_id for item in matching_entries)
        for task_id, (value, row) in _selected_jsonl_rows(
            content, selected_ids, f"BFCL public question object {git_path}"
        ).items():
            rows[task_id] = (value, row, size, digest)

    tasks: list[BfclV4PublicPilotTask] = []
    for roster_item in roster:
        value, row, blob_size, blob_sha256 = rows[roster_item.task_id]
        if set(value) != {"id", "question", "function"}:
            raise BfclV4PublicPilotError("selected BFCL row schema changed")
        functions = value["function"]
        if not isinstance(functions, list) or not functions:
            raise BfclV4PublicPilotError("selected BFCL function schema list is invalid")
        names = tuple(item.get("name") for item in functions if isinstance(item, dict))
        if len(names) != len(functions) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise BfclV4PublicPilotError("selected BFCL function names are invalid")
        tasks.append(
            BfclV4PublicPilotTask(
                task_id=roster_item.task_id,
                category=roster_item.category,
                split=roster_item.split,
                semantic_family=roster_item.semantic_family,
                question_git_path=roster_item.question_git_path,
                question_blob_size=blob_size,
                question_blob_sha256=blob_sha256,
                row_size=len(row),
                row_sha256=sha256_bytes(row),
                candidate_payload_sha256=canonical_sha256(value),
                question_json=canonical_json(value["question"]),
                function_schemas_json=canonical_json(functions),
                official_function_names=names,
            )
        )
    return BfclV4LoadedPublicPilot(manifest=BFCL_V4_PUBLIC_PILOT_MANIFEST, tasks=tuple(tasks))


def _cast_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Mirror pinned BFCL ``_cast_to_openai_type`` for its OpenAI mapping."""

    for key, raw_value in properties.items():
        if not isinstance(raw_value, dict):
            raise BfclV4PublicPilotError(f"function property {key!r} is not an object")
        value = raw_value
        if "type" not in value:
            value["type"] = "string"
        else:
            source_type = value["type"]
            if not isinstance(source_type, str):
                raise BfclV4PublicPilotError(f"function property {key!r} type is not a string")
            if source_type == "float":
                description = value.get("description")
                if not isinstance(description, str):
                    raise BfclV4PublicPilotError("pinned float conversion requires a description")
                value["format"] = "float"
                value["description"] = description + " This is a float type value."
            value["type"] = BFCL_V4_GORILLA_TO_OPENAPI.get(source_type, "string")

        if value["type"] in {"array", "object"}:
            if "properties" in value:
                nested = value["properties"]
                if not isinstance(nested, dict):
                    raise BfclV4PublicPilotError("nested function properties are not an object")
                value["properties"] = _cast_properties(nested)
            elif "items" in value:
                items = value["items"]
                if not isinstance(items, dict) or not isinstance(items.get("type"), str):
                    raise BfclV4PublicPilotError("array/object items lack a pinned source type")
                try:
                    items["type"] = BFCL_V4_GORILLA_TO_OPENAPI[items["type"]]
                except KeyError as error:
                    raise BfclV4PublicPilotError(
                        "items type is absent from GORILLA_TO_OPENAPI"
                    ) from error
                if items["type"] == "array" and "items" in items:
                    nested_items = items["items"]
                    if not isinstance(nested_items, dict) or not isinstance(
                        nested_items.get("type"), str
                    ):
                        raise BfclV4PublicPilotError("nested array items lack a pinned source type")
                    try:
                        nested_items["type"] = BFCL_V4_GORILLA_TO_OPENAPI[nested_items["type"]]
                    except KeyError as error:
                        raise BfclV4PublicPilotError(
                            "nested items type is absent from GORILLA_TO_OPENAPI"
                        ) from error
                elif items["type"] == "object" and "properties" in items:
                    nested = items["properties"]
                    if not isinstance(nested, dict):
                        raise BfclV4PublicPilotError("nested item properties are not an object")
                    items["properties"] = _cast_properties(nested)
    return properties


def adapt_bfcl_v4_openai_completions_tools(
    functions: Sequence[Mapping[str, Any]],
) -> BfclV4OpenAiToolAdapterResult:
    """Reproduce pinned ``convert_to_tool(..., OPENAI_COMPLETIONS)`` fail-closed."""

    if isinstance(functions, (str, bytes)) or not isinstance(functions, Sequence) or not functions:
        raise BfclV4PublicPilotError("BFCL functions must be a non-empty sequence")
    try:
        copied = deepcopy(_strict_json(canonical_json(list(functions)), "BFCL functions"))
    except (TypeError, ValueError) as error:
        raise BfclV4PublicPilotError("BFCL functions are not strict JSON") from error
    converted: list[dict[str, Any]] = []
    bindings: list[BfclV4ToolNameBinding] = []
    seen_official: set[str] = set()
    seen_wire: set[str] = set()
    for raw_item in copied:
        if not isinstance(raw_item, dict):
            raise BfclV4PublicPilotError("BFCL function schema is not an object")
        item = raw_item
        official_name = item.get("name")
        if not isinstance(official_name, str) or not official_name:
            raise BfclV4PublicPilotError("BFCL function name is invalid")
        wire_name = official_name.replace(".", "_")
        if official_name in seen_official:
            raise BfclV4PublicPilotError("official function name repeats")
        if wire_name in seen_wire:
            raise BfclV4PublicPilotError(f"wire-name collision after dot substitution: {wire_name}")
        try:
            binding = BfclV4ToolNameBinding(
                official_name=official_name,
                wire_name=wire_name,
            )
        except ValueError as error:
            raise BfclV4PublicPilotError(
                "BFCL function name cannot be represented on the wire"
            ) from error
        seen_official.add(official_name)
        seen_wire.add(wire_name)
        item["name"] = wire_name
        parameters = item.get("parameters")
        if not isinstance(parameters, dict) or not isinstance(parameters.get("properties"), dict):
            raise BfclV4PublicPilotError("BFCL function parameters/properties are invalid")
        parameters["type"] = "object"
        parameters["properties"] = _cast_properties(parameters["properties"])
        converted.append({"type": "function", "function": item})
        bindings.append(binding)
    return BfclV4OpenAiToolAdapterResult(
        tools_json=canonical_json(converted),
        name_bindings=tuple(bindings),
    )


def adapt_bfcl_v4_public_pilot_task(
    task: BfclV4PublicPilotTask,
) -> BfclV4OpenAiToolAdapterResult:
    """Adapt schemas already bound to a pinned candidate-visible task."""

    checked = BfclV4PublicPilotTask.model_validate(task, strict=True)
    functions = _strict_json(checked.function_schemas_json, "pinned BFCL task functions")
    if not isinstance(functions, list):  # pragma: no cover - task contract already guarantees this
        raise BfclV4PublicPilotError("pinned BFCL task functions are not a list")
    adapted = adapt_bfcl_v4_openai_completions_tools(functions)
    if (
        tuple(item.official_name for item in adapted.name_bindings)
        != checked.official_function_names
    ):
        raise BfclV4PublicPilotError("adapter name roster differs from the pinned task")
    return adapted


def verify_bfcl_v4_adapter_sources(
    checkout: str | Path,
) -> BfclV4AdapterSourceVerification:
    """Bind exact upstream converter/mapping bytes without executing them."""

    try:
        resolved, git = _checkout_and_git(checkout)
    except BfclV4FixtureBridgeError as error:
        raise BfclV4PublicPilotError("pinned BFCL checkout verification failed") from error
    sources: list[BfclV4SourceFileBinding] = []
    for git_path, (size, digest) in sorted(BFCL_V4_PILOT_ADAPTER_SOURCE_IDENTITIES.items()):
        content = _verified_blob(git, resolved, git_path, size, digest)
        sources.append(BfclV4SourceFileBinding(git_path=git_path, size=len(content), sha256=digest))
    frozen = tuple(sources)
    return BfclV4AdapterSourceVerification(
        sources=frozen,
        source_bundle_sha256=canonical_sha256(
            tuple(
                {"git_path": item.git_path, "sha256": item.sha256, "size": item.size}
                for item in frozen
            )
        ),
    )


def normalize_bfcl_v4_tool_calls(
    calls: tuple[BfclV4WireToolCall, ...],
    name_bindings: tuple[BfclV4ToolNameBinding, ...],
) -> str:
    """Recover official names and canonically sort calls, retaining multiplicity."""

    if not isinstance(calls, tuple) or not isinstance(name_bindings, tuple) or not name_bindings:
        raise BfclV4PublicPilotError("normalization requires tuple calls and name bindings")
    checked_calls = tuple(BfclV4WireToolCall.model_validate(item, strict=True) for item in calls)
    checked_bindings = tuple(
        BfclV4ToolNameBinding.model_validate(item, strict=True) for item in name_bindings
    )
    reverse = {item.wire_name: item.official_name for item in checked_bindings}
    if len(reverse) != len(checked_bindings):
        raise BfclV4PublicPilotError("normalizer wire-name bindings are not unique")
    normalized: list[dict[str, Any]] = []
    for call in checked_calls:
        official = reverse.get(call.wire_name)
        if official is None:
            raise BfclV4PublicPilotError(f"unknown BFCL wire function: {call.wire_name}")
        try:
            arguments = _strict_json(call.arguments_json, "BFCL tool arguments")
        except ValueError as error:
            raise BfclV4PublicPilotError("BFCL tool arguments are invalid") from error
        if not isinstance(arguments, dict):
            raise BfclV4PublicPilotError("BFCL tool arguments must be a JSON object")
        normalized.append({"name": official, "arguments": arguments})
    normalized.sort(key=lambda item: (item["name"], canonical_json(item["arguments"])))
    return canonical_json(normalized)


def select_bfcl_v4_pure_at_b_plurality(
    samples: tuple[BfclV4PureAtBSample, ...],
    name_bindings: tuple[BfclV4ToolNameBinding, ...],
) -> BfclV4PureAtBSelection:
    """Select canonical plurality without accepting a grader/target argument."""

    if not isinstance(samples, tuple) or not samples:
        raise BfclV4PublicPilotError("PURE@B requires a non-empty frozen sample tuple")
    checked = tuple(BfclV4PureAtBSample.model_validate(item, strict=True) for item in samples)
    if len({item.sample_id for item in checked}) != len(checked):
        raise BfclV4PublicPilotError("PURE@B sample IDs must not repeat")
    labels: list[str | None] = []
    for sample in checked:
        if sample.calls is None:
            labels.append(None)
            continue
        try:
            labels.append(normalize_bfcl_v4_tool_calls(sample.calls, name_bindings))
        except (BfclV4PublicPilotError, ValueError):
            labels.append(None)
    counts = Counter(label for label in labels if label is not None)
    if not counts:
        selected_index = 0
        selected_label = None
        plurality_count = 0
        tie = False
        all_abstained = True
    else:
        plurality_count = max(counts.values())
        winners = {label for label, count in counts.items() if count == plurality_count}
        selected_index = next(index for index, label in enumerate(labels) if label in winners)
        selected_label = labels[selected_index]
        tie = len(winners) > 1
        all_abstained = False
    selected = checked[selected_index]
    return BfclV4PureAtBSelection(
        selected_sample_id=selected.sample_id,
        selected_frozen_index=selected_index,
        selected_calls=selected.calls,
        normalized_output_json=selected_label,
        plurality_count=plurality_count,
        tie=tie,
        all_abstained=all_abstained,
    )
