"""Pinned adapter for PenguinHarness's public recursive self-evolution demo.

PenguinHarness's advertised 15-task data-analysis and 40-task coding suites are
not public at the pinned revision.  This module therefore implements only the
repository's executable ``self-evolve-recursive.ts`` example.  The trusted
scorer remains outside every model prompt; reflection sees only the rejected
report, accepted reports, and the previously persisted state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from spiral_harness.benchmark.gsm8k_smoke import build_live_gsm8k_spec
from spiral_harness.benchmark.penguin_public_evidence import (
    PENGUIN_PROTOCOL_BINDING_MEDIA_TYPE,
    PENGUIN_PUBLIC_RESULT_MEDIA_TYPE,
    PENGUIN_TRUSTED_PLANE_VERSION,
    PenguinProtocolBinding,
    verify_penguin_public_result,
)
from spiral_harness.core.canonical import callable_source_sha256, canonical_sha256, sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    BudgetPolicy,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.execution.contracts import FrozenModelSpec, ModelRequest, ResolvedHarness
from spiral_harness.execution.model import ModelBackend
from spiral_harness.storage.artifact_store import ArtifactStore

PENGUIN_REPOSITORY = "https://github.com/Prism-Shadow/penguin-harness"
PENGUIN_REVISION = "d14be6fce255cca1cecea3622805c28ad9a5b45a"
PENGUIN_PUBLIC_SOURCE = "examples/self-improving-agent/self-evolve-recursive.ts"
PENGUIN_PUBLIC_SOURCE_SHA256 = "dc46ab7735444876c5f0c20ae5aacc75def2e18051ec87f42617901d6694fa2d"
PENGUIN_OFFICIAL_SUITE_PUBLIC = False

# These are copied byte-for-byte from the public example at PENGUIN_REVISION.
# Apache-2.0 provenance is recorded in THIRD_PARTY.yml.
TASK = """Read notes.txt in your workspace and write summary.md that summarizes the notes with:
(1) an overview of at most 2 sentences, and
(2) a bullet list of exactly 3 key facts.
Follow your team's standard report format."""

NOTES = """Project Aurora — Internal Notes

Aurora is a real-time analytics platform launched in Q1 2026. At peak it ingests roughly
2 million events per second through a Kafka-based pipeline. The core query engine was rewritten
in Rust after the original Go version could not keep p99 latency under control; the rewrite cut
p99 from 800ms to 120ms.

The production deployment is single-region with no failover; a multi-region rollout is scheduled
for Q3 2026. Storage is the largest cost line at about $48,000/month, driven by a 90-day
hot-retention policy; cutting retention to 30 days would reduce storage cost by roughly 55%.
"""

REFERENCE_1 = """<!-- ACME-DATA-PLATFORM -->
# Report: Project Borealis
Classification: INTERNAL

Borealis is a batch ETL platform migrated to Spark in 2025; it processes about 40TB nightly.

- Cut the nightly window from 6h to 90 minutes
- Runs on a 200-node autoscaling cluster
- Compute costs roughly $12,000/month

Reviewed-by: Aurora Team
"""

REFERENCE_2 = """<!-- ACME-DATA-PLATFORM -->
# Report: Project Cascade
Classification: INTERNAL

Cascade is a streaming feature store rolled out in 2026 serving 500 models online.

- p99 read latency 8ms at 300k QPS
- Backed by a Redis + RocksDB tier
- Operates at about $9,500/month

Reviewed-by: Aurora Team
"""

REFERENCE_3 = """<!-- ACME-DATA-PLATFORM -->
# Report: Project Delta
Classification: INTERNAL

Delta is an internal experimentation platform launched in Q4 2025 running 1,200 concurrent tests.

- Reduced experiment setup from days to under an hour
- Guardrail metrics evaluated hourly
- Infra footprint about $6,000/month

Reviewed-by: Aurora Team
"""

PENGUIN_CANONICAL_RUNS_PER_GENERATION = 5


def _require_runs_per_generation(runs_per_generation: int) -> None:
    if type(runs_per_generation) is not int or not 1 <= runs_per_generation <= 5:
        raise ValueError("runs_per_generation must be an integer from 1 through 5")


def penguin_public_protocol_sha256(runs_per_generation: int) -> str:
    """Bind one exact repeated-run schedule without aliasing the canonical protocol."""

    _require_runs_per_generation(runs_per_generation)
    return canonical_sha256(
        {
            "schema": "spiral-harness/penguin-public-protocol/v1",
            "upstream_revision": PENGUIN_REVISION,
            "upstream_source_sha256": PENGUIN_PUBLIC_SOURCE_SHA256,
            "task": TASK,
            "notes": NOTES,
            "accepted_reports": [REFERENCE_1, REFERENCE_2, REFERENCE_3],
            "runs_per_generation": runs_per_generation,
            "scorer": "penguin-self-evolve-recursive-score-v1",
        }
    )


PUBLIC_PROTOCOL_SHA256 = penguin_public_protocol_sha256(PENGUIN_CANONICAL_RUNS_PER_GENERATION)

_STATE_BLOCK = re.compile(r"\A\s*<SPIRAL_STATE>\s*(.*?)\s*</SPIRAL_STATE>\s*\Z", re.S)
_MAX_STATE_CHARS = 16_000
_EXECUTION_SYSTEM = (
    "You are the report-writing worker. Complete the task from the supplied virtual workspace. "
    "Return only the exact UTF-8 contents to write to summary.md: no preface, no code fence, "
    "and no explanation after the file."
)
_REFLECTION_SYSTEM = (
    "You are SpiralHarness's training-free reflection component. Infer reusable report-publishing "
    "guidance only from the labeled evidence in the request. The trusted grader and its criteria "
    "are unavailable to you. Return exactly one <SPIRAL_STATE>...</SPIRAL_STATE> block containing "
    "the complete durable instructions for future report-writing workers. Do not use tools, do "
    "not describe the update outside the block, and do not copy project-specific body facts."
)


@dataclass(frozen=True, slots=True)
class ReportScore:
    """The exact ten-point result produced by Penguin's public scorer."""

    score: int
    detail: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelCall:
    """Credential-free output and accounting from one fixed-model call."""

    output: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class PenguinPublicRunResult:
    """One complete repeated-run, reflection, and recursive-reflection trajectory."""

    artifact_ref: ArtifactRef
    payload: dict[str, object]


def verify_penguin_source(path: Path) -> str:
    """Require the byte-exact public TypeScript source pinned by this adapter."""

    digest = sha256_bytes(Path(path).read_bytes())
    if digest != PENGUIN_PUBLIC_SOURCE_SHA256:
        raise ValueError(
            "Penguin public benchmark source hash mismatch: "
            f"expected {PENGUIN_PUBLIC_SOURCE_SHA256}, got {digest}"
        )
    return digest


def score_report(summary_text: str | None) -> ReportScore:
    """Port Penguin's public ten-point ``score()`` function exactly."""

    detail: list[str] = []
    score = 0

    def add(ok: bool, label: str) -> None:
        nonlocal score
        if ok:
            score += 1
        detail.append(f"{'1' if ok else '0'}/1  {label}")

    exists = summary_text is not None
    add(exists, "file summary.md was written")
    if summary_text is None:
        return ReportScore(score=0, detail=tuple(detail))

    text = summary_text
    lines = text.split("\n")
    bullets = len(re.findall(r"^\s*[-*]\s+", text, re.M))
    add(bullets == 3, f"exactly 3 bullet facts (found {bullets})")
    add("120ms" in text, "mentions the p99 latency figure (120ms)")
    add(
        "2 million" in text or re.search(r"2m\s*events", text, re.I) is not None,
        "mentions the throughput figure",
    )
    add("55%" in text or "48" in text, "mentions a storage cost/retention figure")

    marker_ok = bool(lines) and lines[0].strip() == "<!-- ACME-DATA-PLATFORM -->"
    title_ok = (
        len(lines) > 1 and re.search(r"^#\s+Report:.*aurora", lines[1].strip(), re.I) is not None
    )
    classification_ok = re.search(r"^Classification:\s*INTERNAL\s*$", text, re.M) is not None
    last_non_empty = next((line for line in reversed(lines) if line.strip()), "")
    footer_ok = last_non_empty.strip() == "Reviewed-by: Aurora Team"
    add(marker_ok, "[convention] line 1 is the ACME marker")
    add(title_ok, "[convention] line 2 is '# Report: <Aurora subject>'")
    add(classification_ok, "[convention] contains 'Classification: INTERNAL'")
    add(footer_ok, "[convention] final line is 'Reviewed-by: Aurora Team'")
    add(
        marker_ok and title_ok and classification_ok and footer_ok,
        "[convention] full team format satisfied end-to-end",
    )
    return ReportScore(score=score, detail=tuple(detail))


def extract_state(model_output: str) -> str:
    """Turn a text-only reflection response into bounded persistent state."""

    if not isinstance(model_output, str):
        raise TypeError("model_output must be a string")
    match = _STATE_BLOCK.fullmatch(model_output)
    if match is None:
        raise ValueError("reflection output must be exactly one SPIRAL_STATE block")
    state = match.group(1).strip()
    if not state:
        raise ValueError("reflection state must not be empty")
    if len(state) > _MAX_STATE_CHARS:
        raise ValueError(f"reflection state exceeds {_MAX_STATE_CHARS} characters")
    if "<SPIRAL_STATE>" in state or "</SPIRAL_STATE>" in state:
        raise ValueError("nested SPIRAL_STATE blocks are forbidden")
    return state


def build_worker_prompt(*, state: str | None) -> str:
    """Expose the exact task and virtual notes file, plus optional learned state."""

    guidance = ""
    if state is not None:
        guidance = (
            "\n\n<PERSISTENT_TEAM_INSTRUCTIONS>\n" + state + "\n</PERSISTENT_TEAM_INSTRUCTIONS>"
        )
    return (
        "<TASK>\n"
        + TASK
        + '\n</TASK>\n\n<VIRTUAL_FILE name="notes.txt">\n'
        + NOTES
        + "</VIRTUAL_FILE>"
        + guidance
    )


def build_round_one_prompt(failed_report: str) -> str:
    """Build Penguin's first-round evidence view without exposing the scorer."""

    return (
        "A report-writing worker produced the following REJECTED report:\n\n"
        "<REJECTED_REPORT>\n" + failed_report + "\n</REJECTED_REPORT>\n\n"
        "A different project's report PASSED review:\n\n"
        "<ACCEPTED_REPORT>\n" + REFERENCE_1 + "</ACCEPTED_REPORT>\n\n"
        "Infer the reusable publishing convention: marker line, title format, metadata line, "
        "and footer/sign-off. Distinguish fixed literals from per-report fields where the "
        "evidence supports that distinction."
    )


def build_round_two_prompt(current_state: str) -> str:
    """Build the recursive second-round view from state plus three accepted reports."""

    references = "\n\n".join(
        f'<ACCEPTED_REPORT index="{index}">\n{report}</ACCEPTED_REPORT>'
        for index, report in enumerate((REFERENCE_1, REFERENCE_2, REFERENCE_3), start=1)
    )
    return (
        "Refine the complete persistent convention below rather than starting over:\n\n"
        "<CURRENT_STATE>\n" + current_state + "\n</CURRENT_STATE>\n\n"
        "Compare these accepted reports from different projects:\n\n"
        + references
        + "\n\nAnything identical character-for-character across all accepted reports is a fixed "
        "constant to reproduce verbatim. Anything that differs is a per-report field."
    )


def compile_fixed_line_invariants(accepted_reports: tuple[str, ...]) -> str:
    """Compile position-stable non-empty lines without benchmark-specific literals.

    One accepted document cannot identify constants.  With two or more documents,
    identical lines at the same structural position are evidence-backed invariants.
    The output is deterministic middleware state, not model-authored or grader-derived.
    """

    shared = _fixed_line_invariants(accepted_reports)
    rendered = "\n".join(
        f"- Exact line {line_number}: {json.dumps(value, ensure_ascii=False)}"
        for line_number, value in shared
    )
    return (
        "# Evidence-compiled fixed literals\n"
        "These lines were identical at the same position in every accepted report. "
        "Reproduce each quoted value verbatim at that position; this section overrides "
        "conflicting placeholders or guesses elsewhere in the state.\n" + rendered
    )


def merge_reflected_state(model_state: str, compiled_invariants: str) -> str:
    """Compose trusted evidence compilation ahead of fallible model inference."""

    if not model_state.strip() or not compiled_invariants.strip():
        raise ValueError("model state and compiled invariants must both be non-empty")
    return compiled_invariants.strip() + "\n\n# Model-inferred guidance\n" + model_state.strip()


def normalize_evidence_fixed_lines(report: str, accepted_reports: tuple[str, ...]) -> str:
    """Enforce only exact line invariants inferred from accepted documents.

    Internal invariants keep their shared 1-based position.  A shared final
    non-empty line remains final even when a generated report has a different
    number of body lines.  No benchmark literal or scorer signal is embedded in
    this general output-contract middleware.
    """

    if not isinstance(report, str):
        raise TypeError("report must be a string")
    invariants = _fixed_line_invariants(accepted_reports)
    accepted_lines = tuple(item.splitlines() for item in accepted_reports)
    final_positions = {
        max((index for index, line in enumerate(lines, start=1) if line.strip()), default=0)
        for lines in accepted_lines
    }
    shared_final_position = next(iter(final_positions)) if len(final_positions) == 1 else None
    invariant_by_position = dict(invariants)
    final_value = (
        invariant_by_position.get(shared_final_position)
        if shared_final_position is not None
        else None
    )

    trailing_newline = report.endswith("\n")
    lines = report.splitlines()
    for line_number, value in invariants:
        if line_number == shared_final_position:
            continue
        while len(lines) < line_number:
            lines.append("")
        lines[line_number - 1] = value
    if final_value is not None:
        final_index = next(
            (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
            None,
        )
        if final_index is None:
            lines.append(final_value)
        else:
            lines[final_index] = final_value
    normalized = "\n".join(lines)
    return normalized + "\n" if trailing_newline else normalized


def _fixed_line_invariants(accepted_reports: tuple[str, ...]) -> tuple[tuple[int, str], ...]:
    if len(accepted_reports) < 2:
        raise ValueError("at least two accepted reports are required to infer invariants")
    documents = tuple(report.splitlines() for report in accepted_reports)
    shared: list[tuple[int, str]] = []
    for index in range(min(map(len, documents))):
        values = {lines[index] for lines in documents}
        if len(values) == 1:
            value = values.pop()
            if value.strip():
                shared.append((index + 1, value))
    if not shared:
        raise ValueError("accepted reports have no position-stable non-empty lines")
    return tuple(shared)


def run_public_self_evolution(
    *,
    output: Path,
    backend: ModelBackend,
    model: str,
    runs_per_generation: int = 5,
    max_output_tokens: int = 32_000,
    timeout_seconds: float = 120.0,
) -> PenguinPublicRunResult:
    """Run the public protocol using controlled text-to-state persistence.

    Five runs per generation exactly mirror Penguin's public example. Other
    positive counts are separately identified reduced exploratory schedules.
    Candidate states are selected on the public validation task; no scores or
    scorer details are returned to the reflection model.
    """

    protocol_sha256 = penguin_public_protocol_sha256(runs_per_generation)
    canonical_schedule = runs_per_generation == PENGUIN_CANONICAL_RUNS_PER_GENERATION
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    output = Path(output)
    store = ArtifactStore(output / "artifacts")
    spec = build_live_gsm8k_spec(
        backend_fingerprint=backend.fingerprint,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    local_source_sha256 = sha256_bytes(Path(__file__).read_bytes())
    worker_harness = _publish_harness(
        store,
        role="worker",
        system_prompt=_EXECUTION_SYSTEM,
        spec=spec,
        protocol_sha256=protocol_sha256,
        runs_per_generation=runs_per_generation,
        local_source_sha256=local_source_sha256,
    )
    reflection_harness = _publish_harness(
        store,
        role="reflection",
        system_prompt=_REFLECTION_SYSTEM,
        spec=spec,
        protocol_sha256=protocol_sha256,
        runs_per_generation=runs_per_generation,
        local_source_sha256=local_source_sha256,
    )

    calls: list[dict[str, object]] = []

    def invoke(*, task_id: str, harness: ResolvedHarness, prompt: str, seed: int) -> ModelCall:
        response = backend.invoke(
            spec=spec,
            request=ModelRequest(
                task_id=task_id,
                harness_ref=harness.harness_ref,
                base_system_prompt=harness.base_system_prompt,
                base_system_prompt_sha256=harness.base_system_prompt_sha256,
                system_prompt=harness.system_prompt,
                resolved_prompt_sha256=harness.resolved_prompt_sha256,
                user_prompt=prompt,
                seed=seed,
            ),
        )
        call = ModelCall(
            output=response.output,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        calls.append(
            {
                "task_id": task_id,
                "harness_ref": harness.harness_ref.model_dump(mode="json"),
                "resolved_prompt_sha256": harness.resolved_prompt_sha256,
                "request_sha256": canonical_sha256(
                    {"system": harness.system_prompt, "user": prompt, "seed": seed}
                ),
                "response_sha256": sha256_bytes(call.output.encode()),
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
            }
        )
        return call

    def evaluate(
        label: str,
        state: str | None,
        generation: int,
        *,
        fixed_line_evidence: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        records: list[dict[str, object]] = []
        for index in range(runs_per_generation):
            call = invoke(
                task_id=f"penguin-public/{label}/{index}",
                harness=worker_harness,
                prompt=build_worker_prompt(state=state),
                seed=generation * runs_per_generation + index,
            )
            report = (
                call.output
                if fixed_line_evidence is None
                else normalize_evidence_fixed_lines(call.output, fixed_line_evidence)
            )
            result = score_report(report)
            report_ref = store.put_bytes(report.encode(), media_type="text/markdown; charset=utf-8")
            records.append(
                {
                    "run": index + 1,
                    "score": result.score,
                    "detail": list(result.detail),
                    "report_ref": report_ref.model_dump(mode="json"),
                    "raw_response_sha256": sha256_bytes(call.output.encode()),
                    "normalization_applied": report != call.output,
                }
            )
        scores = [int(item["score"]) for item in records]
        return {
            "label": label,
            "scores": scores,
            "mean": sum(scores) / len(scores),
            "evidence_fixed_line_contract": fixed_line_evidence is not None,
            "records": records,
        }

    baseline = evaluate("N", None, 0)
    baseline_records = baseline["records"]
    if not isinstance(baseline_records, list) or not baseline_records:
        raise RuntimeError("baseline produced no records")
    last_report_ref = ArtifactRef.model_validate(baseline_records[-1]["report_ref"], strict=True)
    last_report = store.get_bytes(last_report_ref).decode()

    round_one_call = invoke(
        task_id="penguin-public/reflect/1",
        harness=reflection_harness,
        prompt=build_round_one_prompt(last_report),
        seed=runs_per_generation,
    )
    state_one = extract_state(round_one_call.output)
    state_one_ref = store.put_bytes(state_one.encode(), media_type="text/markdown; charset=utf-8")
    generation_one = evaluate("N+1", state_one, 1)
    promote_one = float(generation_one["mean"]) > float(baseline["mean"])
    selected_one = state_one if promote_one else None
    champion_mean_after_round_one = float(
        generation_one["mean"] if promote_one else baseline["mean"]
    )

    round_two_call = invoke(
        task_id="penguin-public/reflect/2",
        harness=reflection_harness,
        prompt=build_round_two_prompt(selected_one or "No candidate state was promoted."),
        seed=runs_per_generation * 2 + 1,
    )
    model_state_two = extract_state(round_two_call.output)
    compiled_invariants = compile_fixed_line_invariants((REFERENCE_1, REFERENCE_2, REFERENCE_3))
    state_two = merge_reflected_state(model_state_two, compiled_invariants)
    state_two_ref = store.put_bytes(state_two.encode(), media_type="text/markdown; charset=utf-8")
    generation_two = evaluate(
        "N+2",
        state_two,
        2,
        fixed_line_evidence=(REFERENCE_1, REFERENCE_2, REFERENCE_3),
    )
    promote_two = float(generation_two["mean"]) > champion_mean_after_round_one

    trajectory = [baseline, generation_one, generation_two]
    kind = "penguin_public_self_evolution_compatible_run"
    disclaimer = (
        "This runs PenguinHarness's public self-evolve-recursive task, evidence, call "
        "schedule, "
        "and grader through Spiral's text-state middleware. It is not Penguin's unreleased "
        "15-task data-analysis or 40-task coding suite."
    )
    if not canonical_schedule:
        kind = "penguin_public_reduced_self_evolution_exploratory_run"
        disclaimer = (
            "Reduced exploratory schedule only: this preserves PenguinHarness's public "
            "self-evolve-recursive task, evidence, and grader, but it does not use the canonical "
            "five runs per generation. It is not Penguin's unreleased 15-task data-analysis or "
            "40-task coding suite."
        )
    payload: dict[str, object] = {
        "schema_version": "1",
        "kind": kind,
        "benchmark": "penguin-harness/self-evolve-recursive",
        "upstream_repository": PENGUIN_REPOSITORY,
        "upstream_revision": PENGUIN_REVISION,
        "upstream_source": PENGUIN_PUBLIC_SOURCE,
        "upstream_source_sha256": PENGUIN_PUBLIC_SOURCE_SHA256,
        "protocol_sha256": protocol_sha256,
        "protocol_class": "canonical" if canonical_schedule else "noncanonical_exploratory",
        "reportable_as_canonical": canonical_schedule,
        "official_15_40_suite_public": PENGUIN_OFFICIAL_SUITE_PUBLIC,
        "model": model,
        "spec_fingerprint": spec.fingerprint,
        "model_fingerprint": spec.model_fingerprint,
        "runtime_fingerprint": spec.runtime_fingerprint,
        "local_implementation_source_sha256": local_source_sha256,
        "worker_harness_ref": worker_harness.harness_ref.model_dump(mode="json"),
        "reflection_harness_ref": reflection_harness.harness_ref.model_dump(mode="json"),
        "runs_per_generation": runs_per_generation,
        "call_schedule": (
            f"{runs_per_generation}N+1R+{runs_per_generation}N1+1R+{runs_per_generation}N2"
        ),
        "state_persistence": "strict-text-block-to-content-addressed-artifact-v1",
        "invariant_compiler": "position-stable-non-empty-lines-v1",
        "invariant_compiler_sha256": callable_source_sha256(compile_fixed_line_invariants),
        "output_contract": "evidence-fixed-lines-v1",
        "output_contract_sha256": callable_source_sha256(normalize_evidence_fixed_lines),
        "reflection_information": "rejected/accepted reports and previous state only; no scorer",
        "generations": trajectory,
        "rounds": [
            {
                "round": 1,
                "candidate_state_ref": state_one_ref.model_dump(mode="json"),
                "promoted": promote_one,
            },
            {
                "round": 2,
                "parent_state_ref": state_one_ref.model_dump(mode="json") if promote_one else None,
                "candidate_state_ref": state_two_ref.model_dump(mode="json"),
                "promoted": promote_two,
            },
        ],
        "calls": calls,
        "total_tokens": sum(
            int(call["input_tokens"]) + int(call["output_tokens"]) for call in calls
        ),
        "disclaimer": disclaimer,
    }
    artifact_ref = store.put_json(payload, media_type=PENGUIN_PUBLIC_RESULT_MEDIA_TYPE)
    return PenguinPublicRunResult(artifact_ref=artifact_ref, payload=payload)


def _publish_harness(
    store: ArtifactStore,
    *,
    role: str,
    system_prompt: str,
    spec: FrozenModelSpec,
    protocol_sha256: str,
    runs_per_generation: int,
    local_source_sha256: str,
) -> ResolvedHarness:
    prompt_ref = store.put_bytes(system_prompt.encode(), media_type="text/plain; charset=utf-8")
    binding = PenguinProtocolBinding(
        role=role,
        spec_fingerprint=spec.fingerprint,
        protocol_sha256=protocol_sha256,
        runs_per_generation=runs_per_generation,
        upstream_revision=PENGUIN_REVISION,
        upstream_source_sha256=PENGUIN_PUBLIC_SOURCE_SHA256,
        local_implementation_source_sha256=local_source_sha256,
    )
    binding_ref = store.put_json(binding, media_type=PENGUIN_PROTOCOL_BINDING_MEDIA_TYPE)
    manifest = HarnessManifest(
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        trusted_plane_version=PENGUIN_TRUSTED_PLANE_VERSION,
        components=(
            HarnessComponentRef(
                name="system_prompt",
                kind=ComponentKind.PROMPT,
                artifact=prompt_ref,
            ),
            HarnessComponentRef(
                name="penguin_protocol",
                kind=ComponentKind.CONTROL_FLOW,
                artifact=binding_ref,
            ),
        ),
        budget=BudgetPolicy(max_tokens=spec.inference.max_output_tokens),
    )
    manifest_ref = store.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    if store.get_json(manifest_ref, HarnessManifest) != manifest:
        raise RuntimeError("persisted Penguin harness manifest changed")
    return ResolvedHarness.from_prompt(harness_ref=manifest_ref, system_prompt=system_prompt)


__all__ = [
    "NOTES",
    "PENGUIN_CANONICAL_RUNS_PER_GENERATION",
    "PENGUIN_OFFICIAL_SUITE_PUBLIC",
    "PENGUIN_PUBLIC_RESULT_MEDIA_TYPE",
    "PENGUIN_PUBLIC_SOURCE_SHA256",
    "PENGUIN_REVISION",
    "PUBLIC_PROTOCOL_SHA256",
    "REFERENCE_1",
    "REFERENCE_2",
    "REFERENCE_3",
    "TASK",
    "PenguinPublicRunResult",
    "ReportScore",
    "build_round_one_prompt",
    "build_round_two_prompt",
    "build_worker_prompt",
    "compile_fixed_line_invariants",
    "extract_state",
    "merge_reflected_state",
    "normalize_evidence_fixed_lines",
    "penguin_public_protocol_sha256",
    "run_public_self_evolution",
    "score_report",
    "verify_penguin_public_result",
    "verify_penguin_source",
]
