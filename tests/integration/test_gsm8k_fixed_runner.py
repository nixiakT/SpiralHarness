from __future__ import annotations

import json

from spiral_harness.benchmark.gsm8k import GSM8KBenchmarkAdapter
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    AttemptBudget,
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
    PromptHarness,
)
from spiral_harness.execution.model import (
    FixedModelRunner,
    ReplayBackend,
)
from spiral_harness.storage.artifact_store import ArtifactStore


def test_score_free_fixed_runner_closes_into_trusted_gsm8k_grading(tmp_path) -> None:
    train_rows = (
        {
            "question": "Lina has 2 shells and finds 3 more. How many shells does she have?",
            "answer": "Add the shells.\n#### 5",
        },
        {
            "question": "Lina has 20 shells and finds 30 more. How many shells does she have?",
            "answer": "Add the shells.\n#### 50",
        },
        {
            "question": "Four boxes hold 6 pens each. How many pens are there?",
            "answer": "Multiply.\n#### 24",
        },
        {
            "question": "A class has 11 girls and 9 boys. How many students are there?",
            "answer": "Add the students.\n#### 20",
        },
    )
    test_rows = (
        {
            "question": "A shelf has 7 red and 8 blue books. How many books are there?",
            "answer": "Add the books.\n#### 15",
        },
    )
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    test_path.write_text(
        "".join(json.dumps(row) + "\n" for row in test_rows),
        encoding="utf-8",
    )
    adapter = GSM8KBenchmarkAdapter(train_path, test_path, verify_pinned=False)
    train_task_ids = (
        *adapter.task_roster(ProtocolPartition.EXPLORATION),
        *adapter.task_roster(ProtocolPartition.GATE),
    )
    task = next(
        adapter.load_task(task_id)
        for task_id in train_task_ids
        if adapter.load_task(task_id).question.startswith("Lina has 2 shells")
    )

    backend = ReplayBackend(
        fingerprint="replay-backend:fixture-v1",
        default_response=BackendResponse(
            output="The answer is 2 + 3.\n#### 5",
            usage=BackendTokenUsage(input_tokens=24, output_tokens=10),
            cost_usd=0.001,
        ),
    )
    inference = InferenceConfig(
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=32,
        timeout_seconds=10.0,
    )
    spec = FrozenModelSpec(
        backend="deterministic-replay",
        backend_fingerprint=backend.fingerprint,
        model="fixture/model",
        revision="snapshot-2026-08-11",
        tokenizer="fixture/tokenizer",
        tokenizer_revision="snapshot-2026-08-11",
        runtime="python-3.12/replay-v1",
        inference=inference,
    )
    store = ArtifactStore(tmp_path / "artifacts")
    ledger = AttemptLedger(
        store,
        ledger_id="gsm8k-fixed-runner-integration",
        budget=AttemptBudget(
            max_attempts=2,
            max_total_tokens=128,
            max_tokens_per_attempt=64,
        ),
    )
    clock_values = iter((10.0, 10.01, 20.0, 20.02))
    runner = FixedModelRunner(
        spec=spec,
        backend=backend,
        attempt_ledger=ledger,
        clock=lambda: next(clock_values),
    )

    parent = runner.execute(
        task,
        harness=PromptHarness.from_prompt(
            harness_id="parent",
            system_prompt="Solve and end with #### <number>.",
        ),
        seed=17,
    )
    candidate = runner.execute(
        task,
        harness=PromptHarness.from_prompt(
            harness_id="candidate",
            system_prompt="Solve, verify, and end with #### <number>.",
        ),
        seed=17,
    )
    parent_observation = adapter.grade(
        task,
        parent,
        harness_id="parent",
        seed=17,
        execution_fingerprint=parent.execution_fingerprint,
    )
    candidate_observation = adapter.grade(
        task,
        candidate,
        harness_id="candidate",
        seed=17,
        execution_fingerprint=candidate.execution_fingerprint,
    )

    assert parent_observation.score == candidate_observation.score == 1.0
    assert parent.execution_fingerprint == candidate.execution_fingerprint
    assert parent.request_sha256 != candidate.request_sha256
    assert parent.cost_usd == candidate.cost_usd == 0.001
    assert "score" not in type(parent).model_fields
    assert "answer" not in type(parent.task).model_fields
    assert ledger.state().attempts_used == ledger.state().completed_attempts == 2
    assert ledger.state().charged_tokens == 68
