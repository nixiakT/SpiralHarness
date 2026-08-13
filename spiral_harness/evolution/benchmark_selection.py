"""Fail-closed champion selection from validation-only benchmark artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    artifact_paths: tuple[Path, ...], *, output: Path, minimum_gain: float = 0.0
) -> dict[str, object]:
    """Select by validation accuracy, then tokens; test artifacts are forbidden."""

    if len(artifact_paths) < 2:
        raise ValueError("selection requires at least two candidate artifacts")
    candidates = []
    benchmark = model = None
    for path in artifact_paths:
        import json

        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        if payload.get("split") != "val":
            raise ValueError("self-evolution selection accepts validation artifacts only")
        if benchmark is None:
            benchmark, model = payload.get("benchmark"), payload.get("model")
        if payload.get("benchmark") != benchmark or payload.get("model") != model:
            raise ValueError("candidate benchmark and model coordinates must match")
        candidates.append(
            CandidateResult(
                name=f"{payload.get('arm')}@k={payload.get('retrieval_k')}",
                artifact_sha256=canonical_sha256(payload),
                accuracy=float(payload["accuracy"]),
                total_tokens=int(payload["total_tokens"]),
            )
        )
    ranked = sorted(candidates, key=lambda item: (-item.accuracy, item.total_tokens, item.name))
    baseline = next((item for item in candidates if item.name.startswith("pure@")), candidates[0])
    champion = ranked[0]
    gain = champion.accuracy - baseline.accuracy
    accepted = gain >= minimum_gain and champion != baseline
    trace: dict[str, object] = {
        "schema_version": "1",
        "benchmark": benchmark,
        "model": model,
        "selection_partition": "val",
        "objective": "maximize_accuracy_then_minimize_tokens",
        "minimum_gain": minimum_gain,
        "baseline": asdict(baseline),
        "candidates": [asdict(item) for item in candidates],
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
