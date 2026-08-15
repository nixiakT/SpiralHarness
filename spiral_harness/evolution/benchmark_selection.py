"""Fail-closed champion selection from validation-only benchmark artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.storage.artifact_store import ArtifactStore

EVOLUTION_TRACE_MEDIA_TYPE = "application/vnd.spiral-harness.benchmark-evolution-trace.v1+json"


@dataclass(frozen=True, slots=True)
class CandidateResult:
    name: str
    artifact_sha256: str
    accuracy: float
    total_tokens: int


def select_validation_champion(
    artifact_paths: tuple[Path, ...],
    *,
    parent_artifact_path: Path,
    output: Path,
    minimum_gain: float = 0.0,
) -> dict[str, object]:
    """Promote the best validation challenger only when it improves its exact parent.

    Token usage breaks ties between challengers, but never turns an accuracy tie
    with the current parent into a promotion.  The parent is explicit so a later
    round cannot silently fall back to PURE after the champion has changed.
    """

    if len(artifact_paths) < 2:
        raise ValueError("selection requires at least two candidate artifacts")
    if isinstance(minimum_gain, bool) or not isinstance(minimum_gain, (int, float)):
        raise TypeError("minimum_gain must be a finite non-negative number")
    minimum_gain = float(minimum_gain)
    if not isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("minimum_gain must be a finite non-negative number")

    resolved_paths = tuple(Path(path).resolve(strict=True) for path in artifact_paths)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("selection artifact paths must be unique")
    resolved_parent_path = Path(parent_artifact_path).resolve(strict=True)
    parent_indexes = tuple(
        index for index, path in enumerate(resolved_paths) if path == resolved_parent_path
    )
    if len(parent_indexes) != 1:
        raise ValueError("parent artifact must appear exactly once in candidate artifacts")

    candidates: list[CandidateResult] = []
    benchmark = model = None
    for path in resolved_paths:
        import json

        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("split") != "val":
            raise ValueError("self-evolution selection accepts validation artifacts only")
        if benchmark is None:
            benchmark, model = payload.get("benchmark"), payload.get("model")
        if payload.get("benchmark") != benchmark or payload.get("model") != model:
            raise ValueError("candidate benchmark and model coordinates must match")
        accuracy_value = payload["accuracy"]
        if isinstance(accuracy_value, bool) or not isinstance(accuracy_value, (int, float)):
            raise ValueError("candidate accuracy must be an integer or float")
        accuracy = float(accuracy_value)
        if not isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError("candidate accuracy must be finite and within [0, 1]")
        total_tokens = payload["total_tokens"]
        if type(total_tokens) is not int or total_tokens < 0:
            raise ValueError("candidate total_tokens must be a non-negative integer")
        candidate = CandidateResult(
            name=f"{payload.get('arm')}@k={payload.get('retrieval_k')}",
            artifact_sha256=canonical_sha256(payload),
            accuracy=accuracy,
            total_tokens=total_tokens,
        )
        candidates.append(candidate)

    parent_index = parent_indexes[0]
    parent = candidates[parent_index]
    challengers = tuple(
        candidate for index, candidate in enumerate(candidates) if index != parent_index
    )
    challenger = min(
        challengers,
        key=lambda item: (-item.accuracy, item.total_tokens, item.name),
    )
    gain = challenger.accuracy - parent.accuracy
    accepted = gain > 0.0 and gain >= minimum_gain
    champion = challenger if accepted else parent
    trace: dict[str, object] = {
        "schema_version": "1",
        "benchmark": benchmark,
        "model": model,
        "selection_partition": "val",
        "objective": "improve_parent_accuracy_then_minimize_challenger_tokens",
        "minimum_gain": minimum_gain,
        # Retain the legacy field as an exact alias, while making its corrected
        # current-parent meaning explicit for new consumers.
        "baseline": asdict(parent),
        "parent": asdict(parent),
        "candidates": [asdict(item) for item in candidates],
        "challenger": asdict(challenger),
        "champion": asdict(champion),
        "gain": gain,
        "accepted": accepted,
        "rejected": [item.name for item in candidates if item != champion],
    }
    store = ArtifactStore(Path(output) / "artifacts")
    ref = store.put_json(trace, media_type=EVOLUTION_TRACE_MEDIA_TYPE)
    trace["artifact_sha256"] = ref.sha256
    return trace


__all__ = ["EVOLUTION_TRACE_MEDIA_TYPE", "CandidateResult", "select_validation_champion"]
