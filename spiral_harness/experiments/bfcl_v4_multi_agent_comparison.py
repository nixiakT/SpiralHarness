"""Matched bare-model and multi-agent comparison on fresh BFCL v2 development tasks."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from spiral_harness.agents.contracts import (
    MULTI_AGENT_RUN_MEDIA_TYPE,
    AgentSpec,
    AgentTask,
    AgentTeamManifest,
    AgentTurn,
    AgentTurnRequest,
    AgentWorkflow,
    MultiAgentRun,
)
from spiral_harness.agents.orchestration import MultiAgentHarnessRunner, MultiAgentRunError
from spiral_harness.agents.receipt_backend import ReceiptBackedAgentBackend
from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    aggregate_bfcl_v4_public_development_v2_pure_at_b,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4PublicDevelopmentV2Split,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_development_grader import (
    BFCL_V4_PUBLIC_V2_DEVELOPMENT_GRADE_MEDIA_TYPE,
    grade_bfcl_v4_public_v2_development_response,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader import (
    open_bfcl_v4_public_v2_trusted_grader,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    AttemptBudget,
    CandidateTask,
    ExecutionStatus,
    FrozenModelSpec,
    ModelExecutionRecord,
    ResolvedHarness,
)
from spiral_harness.execution.model import FixedModelRunner, ModelBackend
from spiral_harness.experiments.bfcl_v4_public_v2_response_canonical import (
    BfclV4PublicV2CanonicalResponseError,
    parse_bfcl_v4_public_v2_canonical_response,
)
from spiral_harness.storage.artifact_store import ArtifactStore

BFCL_MULTI_AGENT_COMPARISON_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-multi-agent-comparison.v1+json"
)
BFCL_MULTI_AGENT_COMPARISON_PROTOCOL = "bfcl-v4-public-v2-fresh-fit-gate-ma-v1"
BFCL_MULTI_AGENT_TASK_REFS = (
    "fit-00",
    "fit-01",
    "fit-02",
    "fit-03",
    "fit-04",
    "gate-00",
    "gate-01",
    "gate-02",
    "gate-03",
)
BFCL_MULTI_AGENT_TURN_IDS = ("analysis", "critique", "final")
_BASELINE_SYSTEM_PROMPT = """/no_think
You solve Berkeley Function Calling tasks.
Read the user messages and available function schemas. Return only one JSON array.
Each array item must contain exactly two keys: "function_name" and "arguments".
"function_name" must be an available function name and "arguments" must be one JSON object.
Preserve the required call order. Do not use Markdown or add explanation."""
_AGENT_INSTRUCTIONS = {
    "analyst": """/no_think
Analyze one Berkeley Function Calling task. Produce a concrete draft JSON call
array using only the supplied function schemas. Check names, argument types, required fields,
values, and call order. You may briefly explain your reasoning before the draft.""",
    "critic": """/no_think
Audit the analyst's BFCL draft against the original user messages and function
schemas. Identify wrong function choices, missing or invented calls, invalid argument names,
types, values, or ordering. Give the coordinator explicit corrections.""",
    "coordinator": """/no_think
Resolve the BFCL task using the analyst draft and critic audit. Return only
one JSON array. Each item must contain exactly "function_name" and "arguments"; arguments must
be a JSON object. Use only supplied function names, preserve call order, and emit no Markdown,
reasoning, labels, or surrounding text.""",
}


class BfclMultiAgentTaskOutcome(ImmutableModel):
    """Opened public-development outcomes for one frozen comparison task."""

    task_ref: NonEmptyStr
    task_id: NonEmptyStr
    category: NonEmptyStr
    split: Literal["fit", "gate"]
    paired_seeds: tuple[int, int, int]
    baseline_execution_refs: tuple[ArtifactRef, ArtifactRef, ArtifactRef]
    baseline_canonical_responses: tuple[str | None, str | None, str | None]
    baseline_single_correct: bool
    baseline_budget_matched_response: str | None
    baseline_budget_matched_correct: bool
    harness_execution_refs: tuple[ArtifactRef, ArtifactRef, ArtifactRef]
    harness_run_ref: ArtifactRef
    harness_canonical_response: str | None
    harness_correct: bool
    baseline_single_grade_ref: ArtifactRef | None
    baseline_budget_matched_grade_ref: ArtifactRef | None
    harness_grade_ref: ArtifactRef | None


class BfclMultiAgentComparisonResult(ImmutableModel):
    """Complete non-reportable comparison with immutable execution references."""

    schema_version: Literal["1"] = "1"
    protocol_id: Literal["bfcl-v4-public-v2-fresh-fit-gate-ma-v1"] = (
        BFCL_MULTI_AGENT_COMPARISON_PROTOCOL
    )
    model_spec: FrozenModelSpec
    roster_manifest_fingerprint: Sha256
    task_refs: tuple[NonEmptyStr, ...]
    task_roster_fingerprint: Sha256
    root_seed: Annotated[int, Field(ge=0, strict=True)]
    team_fingerprint: Sha256
    workflow_fingerprint: Sha256
    scheduled_model_calls: Literal[54] = 54
    consumed_model_calls: Literal[54] = 54
    total_reported_tokens: Annotated[int, Field(ge=0, strict=True)]
    baseline_single_correct: Annotated[int, Field(ge=0, le=9, strict=True)]
    baseline_budget_matched_correct: Annotated[int, Field(ge=0, le=9, strict=True)]
    harness_correct: Annotated[int, Field(ge=0, le=9, strict=True)]
    outcomes: Annotated[tuple[BfclMultiAgentTaskOutcome, ...], Field(min_length=9, max_length=9)]
    holdout_opened: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    provider_weights_attested: Literal[False] = False
    reportable_result: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def paired_comparison_seed(*, root_seed: int, task_id: str, turn_id: str) -> int:
    """Derive the same three provider seeds for baseline samples and agent turns."""

    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    if turn_id not in BFCL_MULTI_AGENT_TURN_IDS:
        raise ValueError("turn_id is outside the frozen three-call schedule")
    digest = canonical_sha256(
        {
            "domain": "spiral-harness/bfcl-v4-multi-agent-paired-seed/v1",
            "root_seed": root_seed,
            "task_id": task_id,
            "turn_id": turn_id,
        }
    )
    return int(digest[:16], 16) & ((1 << 31) - 1)


def canonicalize_bfcl_candidate_text(output: str) -> str:
    """Project one untrusted text response through a frozen label-free parser."""

    if type(output) is not str or not output.strip():
        raise BfclV4PublicV2CanonicalResponseError("candidate response is empty")
    text = output.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0] in {"```", "```json", "```JSON"}:
            text = "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    candidates: list[str] = []
    for index, character in enumerate(text):
        if character != "[":
            continue
        try:
            value, _end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if type(value) is not list:
            continue
        try:
            canonical = canonical_json(value)
            parse_bfcl_v4_public_v2_canonical_response(canonical)
        except (TypeError, ValueError, BfclV4PublicV2CanonicalResponseError):
            continue
        candidates.append(canonical)
    if len(candidates) != 1:
        raise BfclV4PublicV2CanonicalResponseError(
            "candidate response must contain exactly one valid BFCL call array"
        )
    return candidates[0]


def _task_payload(task) -> str:
    return canonical_json(
        {
            "function_schemas": json.loads(task.function_schemas_json),
            "output_contract": [
                {"arguments": "JSON object", "function_name": "available function name"}
            ],
            "question": json.loads(task.question_json),
        }
    )


def _team(spec: FrozenModelSpec) -> AgentTeamManifest:
    roles = {"analyst": "analysis", "critic": "critique", "coordinator": "synthesis"}
    return AgentTeamManifest(
        team_id="bfcl-analyst-critic-coordinator-v1",
        coordinator_id="coordinator",
        agents=tuple(
            AgentSpec(
                agent_id=agent_id,
                role=roles[agent_id],
                instruction=_AGENT_INSTRUCTIONS[agent_id],
                model_fingerprint=spec.model_fingerprint,
                runtime_fingerprint=spec.runtime_fingerprint,
            )
            for agent_id in ("analyst", "critic", "coordinator")
        ),
        max_turns=3,
        max_context_bytes=524_288,
        max_output_bytes_per_turn=65_536,
        max_total_output_bytes=196_608,
        max_total_tokens=98_304,
    )


def _workflow(team: AgentTeamManifest) -> AgentWorkflow:
    return AgentWorkflow(
        workflow_id="bfcl-analyze-critique-synthesize-v1",
        team_fingerprint=team.fingerprint,
        turns=(
            AgentTurn(
                turn_id="analysis",
                agent_id="analyst",
                objective="Derive a complete candidate function-call array.",
            ),
            AgentTurn(
                turn_id="critique",
                agent_id="critic",
                objective="Audit the candidate against the task and schemas.",
                depends_on=("analysis",),
            ),
            AgentTurn(
                turn_id="final",
                agent_id="coordinator",
                objective="Return the corrected canonical function-call array only.",
                depends_on=("analysis", "critique"),
            ),
        ),
        final_turn_id="final",
    )


def _put_baseline_harness(store: ArtifactStore) -> ResolvedHarness:
    ref = store.put_json(
        {
            "domain": "spiral-harness/bfcl-v4-bare-baseline-prompt/v1",
            "system_prompt": _BASELINE_SYSTEM_PROMPT,
        },
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    return ResolvedHarness.from_prompt(harness_ref=ref, system_prompt=_BASELINE_SYSTEM_PROMPT)


def _parsed_output(record: ModelExecutionRecord) -> str | None:
    execution = record.execution
    if execution.status is not ExecutionStatus.COMPLETED or execution.output is None:
        return None
    try:
        return canonicalize_bfcl_candidate_text(execution.output)
    except BfclV4PublicV2CanonicalResponseError:
        return None


def _grade(
    *,
    store: ArtifactStore,
    grader,
    task_ref: str,
    response: str | None,
) -> tuple[bool, ArtifactRef | None]:
    if response is None:
        return False, None
    receipt = grade_bfcl_v4_public_v2_development_response(
        grader=grader,
        task_ref=task_ref,
        canonical_response=response,
    )
    ref = store.put_json(
        receipt,
        media_type=BFCL_V4_PUBLIC_V2_DEVELOPMENT_GRADE_MEDIA_TYPE,
    )
    return receipt.correct, ref


def _require_empty_output(output: Path) -> None:
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise ValueError("output must be absent or an empty non-symlink directory")
    else:
        output.mkdir(parents=True)


def comparison_protocol_summary(checkout: Path) -> dict[str, object]:
    """Validate and describe the fresh score-free schedule without opening answers."""

    loaded = load_bfcl_v4_public_development_v2(checkout)
    selected = tuple(
        (entry, task)
        for entry, task in zip(loaded.manifest.roster, loaded.tasks, strict=True)
        if entry.split
        in {
            BfclV4PublicDevelopmentV2Split.FIT,
            BfclV4PublicDevelopmentV2Split.GATE,
        }
    )
    expected_splits = ("fit",) * 5 + ("gate",) * 4
    if tuple(entry.split.value for entry, _task in selected) != expected_splits:
        raise ValueError("fresh comparison roster differs from the frozen task refs")
    return {
        "arms": ["baseline-single", "baseline-budget-matched", "multi-agent-harness"],
        "baseline_model_calls": 27,
        "harness_model_calls": 27,
        "hidden_test_evidence": False,
        "holdout_opened": False,
        "model_calls": 54,
        "protocol_id": BFCL_MULTI_AGENT_COMPARISON_PROTOCOL,
        "reportable_result": False,
        "roster_manifest_fingerprint": loaded.manifest.fingerprint,
        "task_ids": [task.task_id for _entry, task in selected],
        "task_refs": list(BFCL_MULTI_AGENT_TASK_REFS),
    }


def run_bfcl_v4_multi_agent_comparison(
    *,
    checkout: Path,
    output: Path,
    spec: FrozenModelSpec,
    backend: ModelBackend,
    root_seed: int = 2026082001,
) -> tuple[BfclMultiAgentComparisonResult, ArtifactRef]:
    """Execute all 54 calls before opening any FIT/GATE Boolean grades."""

    protocol = comparison_protocol_summary(checkout)
    _require_empty_output(output)
    store = ArtifactStore(output / "artifacts")
    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    if not isinstance(backend, ModelBackend):
        raise TypeError("backend must implement ModelBackend")
    if backend.fingerprint != checked_spec.backend_fingerprint:
        raise ValueError("backend differs from the frozen model spec")
    ledger = AttemptLedger(
        store,
        ledger_id=f"{BFCL_MULTI_AGENT_COMPARISON_PROTOCOL}/{root_seed}",
        budget=AttemptBudget(
            max_attempts=54,
            max_total_tokens=54 * 32_768,
            max_tokens_per_attempt=32_768,
        ),
    )
    fixed_runner = FixedModelRunner(spec=checked_spec, backend=backend, attempt_ledger=ledger)
    baseline_harness = _put_baseline_harness(store)
    team = _team(checked_spec)
    workflow = _workflow(team)
    agents_by_id = {agent.agent_id: agent for agent in team.agents}

    def seed_resolver(request: AgentTurnRequest) -> int:
        return paired_comparison_seed(
            root_seed=root_seed,
            task_id=request.task_id,
            turn_id=request.turn_id,
        )

    agent_backends = {
        agent_id: ReceiptBackedAgentBackend(
            agent=agents_by_id[agent_id],
            runner=fixed_runner,
            seed_resolver=seed_resolver,
        )
        for agent_id in agents_by_id
    }
    harness_runner = MultiAgentHarnessRunner(
        team=team,
        workflow=workflow,
        backends=agent_backends,
    )
    loaded = load_bfcl_v4_public_development_v2(checkout)
    selected = tuple(
        (entry, task)
        for entry, task in zip(loaded.manifest.roster, loaded.tasks, strict=True)
        if entry.split
        in {
            BfclV4PublicDevelopmentV2Split.FIT,
            BfclV4PublicDevelopmentV2Split.GATE,
        }
    )
    score_free: list[
        tuple[
            str,
            object,
            tuple[int, int, int],
            tuple[ModelExecutionRecord, ModelExecutionRecord, ModelExecutionRecord],
            tuple[str | None, str | None, str | None],
            MultiAgentRun | None,
            ArtifactRef | None,
            tuple[ArtifactRef, ArtifactRef, ArtifactRef] | None,
            str | None,
        ]
    ] = []
    for task_ref, (_entry, task) in zip(BFCL_MULTI_AGENT_TASK_REFS, selected, strict=True):
        seeds = tuple(
            paired_comparison_seed(root_seed=root_seed, task_id=task.task_id, turn_id=turn_id)
            for turn_id in BFCL_MULTI_AGENT_TURN_IDS
        )
        candidate_task = CandidateTask(task_id=task.task_id, question=_task_payload(task))
        baseline_records = tuple(
            fixed_runner.execute_record(candidate_task, harness=baseline_harness, seed=seed)
            for seed in seeds
        )
        baseline_responses = tuple(_parsed_output(record) for record in baseline_records)
        harness_run: MultiAgentRun | None = None
        harness_run_ref: ArtifactRef | None = None
        harness_execution_refs: tuple[ArtifactRef, ArtifactRef, ArtifactRef] | None = None
        harness_response: str | None = None
        try:
            harness_run = harness_runner.execute(
                AgentTask(task_id=task.task_id, payload=candidate_task.question)
            )
            harness_run_ref = store.put_json(
                harness_run,
                media_type=MULTI_AGENT_RUN_MEDIA_TYPE,
            )
            harness_execution_refs = tuple(
                agent_backends[agent_id].records[-1][1].execution_ref
                for agent_id in ("analyst", "critic", "coordinator")
            )
            with suppress(BfclV4PublicV2CanonicalResponseError):
                harness_response = canonicalize_bfcl_candidate_text(harness_run.final_output)
        except MultiAgentRunError:
            pass
        score_free.append(
            (
                task_ref,
                task,
                seeds,
                baseline_records,
                baseline_responses,
                harness_run,
                harness_run_ref,
                harness_execution_refs,
                harness_response,
            )
        )

    state = ledger.state()
    if state.attempts_used != 54:
        raise RuntimeError("comparison did not consume the exact frozen 54-call schedule")
    grader = open_bfcl_v4_public_v2_trusted_grader(checkout)
    outcomes: list[BfclMultiAgentTaskOutcome] = []
    for (
        task_ref,
        task_value,
        seeds,
        baseline_records,
        baseline_responses,
        _harness_run,
        harness_run_ref,
        harness_execution_refs,
        harness_response,
    ) in score_free:
        if harness_run_ref is None or harness_execution_refs is None:
            raise RuntimeError("comparison lacks one complete three-turn harness execution")
        task = task_value
        baseline_single = baseline_responses[0]
        baseline_single_correct, baseline_single_grade_ref = _grade(
            store=store,
            grader=grader,
            task_ref=task_ref,
            response=baseline_single,
        )
        aggregate = aggregate_bfcl_v4_public_development_v2_pure_at_b(baseline_responses)
        baseline_budget_response = aggregate.selected_canonical_response
        baseline_budget_correct, baseline_budget_grade_ref = _grade(
            store=store,
            grader=grader,
            task_ref=task_ref,
            response=baseline_budget_response,
        )
        harness_correct, harness_grade_ref = _grade(
            store=store,
            grader=grader,
            task_ref=task_ref,
            response=harness_response,
        )
        outcomes.append(
            BfclMultiAgentTaskOutcome(
                task_ref=task_ref,
                task_id=task.task_id,
                category=task.category,
                split=task_ref.split("-", maxsplit=1)[0],
                paired_seeds=seeds,
                baseline_execution_refs=tuple(record.execution_ref for record in baseline_records),
                baseline_canonical_responses=baseline_responses,
                baseline_single_correct=baseline_single_correct,
                baseline_budget_matched_response=baseline_budget_response,
                baseline_budget_matched_correct=baseline_budget_correct,
                harness_execution_refs=harness_execution_refs,
                harness_run_ref=harness_run_ref,
                harness_canonical_response=harness_response,
                harness_correct=harness_correct,
                baseline_single_grade_ref=baseline_single_grade_ref,
                baseline_budget_matched_grade_ref=baseline_budget_grade_ref,
                harness_grade_ref=harness_grade_ref,
            )
        )
    result = BfclMultiAgentComparisonResult(
        model_spec=checked_spec,
        roster_manifest_fingerprint=protocol["roster_manifest_fingerprint"],
        task_refs=BFCL_MULTI_AGENT_TASK_REFS,
        task_roster_fingerprint=canonical_sha256(protocol["task_ids"]),
        root_seed=root_seed,
        team_fingerprint=team.fingerprint,
        workflow_fingerprint=workflow.fingerprint,
        consumed_model_calls=state.attempts_used,
        total_reported_tokens=state.charged_tokens,
        baseline_single_correct=sum(item.baseline_single_correct for item in outcomes),
        baseline_budget_matched_correct=sum(
            item.baseline_budget_matched_correct for item in outcomes
        ),
        harness_correct=sum(item.harness_correct for item in outcomes),
        outcomes=tuple(outcomes),
    )
    result_ref = store.put_json(result, media_type=BFCL_MULTI_AGENT_COMPARISON_MEDIA_TYPE)
    terminal = {
        "baseline_budget_matched": f"{result.baseline_budget_matched_correct}/9",
        "baseline_single": f"{result.baseline_single_correct}/9",
        "consumed_model_calls": result.consumed_model_calls,
        "harness": f"{result.harness_correct}/9",
        "protocol_id": result.protocol_id,
        "reportable_result": False,
        "result_ref": result_ref.model_dump(mode="json"),
        "total_reported_tokens": result.total_reported_tokens,
    }
    (output / "terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result, result_ref


__all__ = [
    "BFCL_MULTI_AGENT_COMPARISON_MEDIA_TYPE",
    "BFCL_MULTI_AGENT_COMPARISON_PROTOCOL",
    "BFCL_MULTI_AGENT_TASK_REFS",
    "BFCL_MULTI_AGENT_TURN_IDS",
    "BfclMultiAgentComparisonResult",
    "BfclMultiAgentTaskOutcome",
    "canonicalize_bfcl_candidate_text",
    "comparison_protocol_summary",
    "paired_comparison_seed",
    "run_bfcl_v4_multi_agent_comparison",
]
