from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.gsm8k as gsm8k_module
from spiral_harness.benchmark.base import BenchmarkAdapter
from spiral_harness.benchmark.gsm8k import (
    DEFAULT_GSM8K_SPLIT_CONFIG,
    GSM8K_ANSWER_PATTERN,
    GSM8K_PINNED_DEFAULT_ADAPTER_FINGERPRINT,
    GSM8K_PINNED_DEFAULT_PARTITION_COUNTS,
    GSM8K_PINNED_DEFAULT_SPLIT_FINGERPRINT,
    GSM8KBenchmarkAdapter,
    GSM8KDataError,
    GSM8KGradingError,
    GSM8KPartitionManifest,
    GSM8KSplitConfig,
    GSM8KSplitManifest,
    GSM8KTask,
    GSM8KTaskManifestEntry,
    extract_gsm8k_answer,
    gsm8k_template_family_sha256,
    load_gsm8k_jsonl,
    normalize_gsm8k_template,
)
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.verification.models import TrialStatus

TRAIN_ROWS = (
    {
        "question": "Mia has 3 apples and buys 2 more. How many apples does she have?",
        "answer": "Mia adds them. 3 + 2 = 5.\n#### 5",
    },
    {
        "question": "Mia has 30 apples and buys 20 more. How many apples does she have?",
        "answer": "Mia adds them. 30 + 20 = 50.\n#### 50",
    },
    {
        "question": "A bus seats 12 people in each of 4 rows. How many people fit?",
        "answer": "Multiply.\n#### 48",
    },
    {
        "question": "Kai saved $15 and then saved $6. What is the total?",
        "answer": "Add the savings.\n#### 21",
    },
    {
        "question": "There are 100 birds and 27 fly away. How many remain?",
        "answer": "Subtract.\n#### 73",
    },
    {
        "question": "A 2.5 meter ribbon is joined to a 1.5 meter ribbon. What is the length?",
        "answer": "Add the lengths.\n#### 4.0",
    },
)

TEST_ROWS = (
    {
        "question": "Nia earns $1,200 and spends $250. How much remains?",
        "answer": "Subtract.\n#### 950",
    },
    {
        "question": "The temperature is 4 degrees and falls by 9. What is it now?",
        "answer": "Subtract.\n#### -5",
    },
)


def write_jsonl(path: Path, rows: tuple[dict[str, str], ...]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def gsm8k_files(tmp_path: Path) -> tuple[Path, Path]:
    return (
        write_jsonl(tmp_path / "train.jsonl", TRAIN_ROWS),
        write_jsonl(tmp_path / "test.jsonl", TEST_ROWS),
    )


@pytest.fixture
def adapter(gsm8k_files: tuple[Path, Path]) -> GSM8KBenchmarkAdapter:
    return GSM8KBenchmarkAdapter(*gsm8k_files, verify_pinned=False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\n#### -12", "-12"),
        ("work\n#### 12.50", "12.50"),
        ("work\n#### 1,234,567", "1234567"),
        ("#### -1,234.50", "-1234.50"),
        ("#### 7 then correction #### 8", "7"),
        ("the odd official spelling is #### 1..2", "1..2"),
        ("#### 42 extra words", "42"),
        ("####42", None),
        ("answer: 42", None),
        ("#### +42", None),
    ],
)
def test_official_answer_extraction_golden_cases(text: str, expected: str | None) -> None:
    assert GSM8K_ANSWER_PATTERN == r"#### (\-?[0-9\.\,]+)"
    assert extract_gsm8k_answer(text) == expected


def test_answer_extraction_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        extract_gsm8k_answer(42)  # type: ignore[arg-type]


def test_strict_jsonl_loader_accepts_only_exact_valid_schema(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "valid.jsonl", TRAIN_ROWS[:2])

    examples = load_gsm8k_jsonl(path)

    assert examples[0].question == TRAIN_ROWS[0]["question"]
    assert examples[0].answer.endswith("#### 5")
    assert len(examples) == 2


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "\n",
        "not json\n",
        "[]\n",
        '{"question":"q"}\n',
        '{"question":"q","answer":"#### 1","extra":true}\n',
        '{"question":1,"answer":"#### 1"}\n',
        '{"question":"q","answer":""}\n',
        '{"question":" q","answer":"#### 1"}\n',
        '{"question":"q","answer":"there is no marker"}\n',
        '{"question":"q","question":"other","answer":"#### 1"}\n',
        ('{"question":"same","answer":"#### 1"}\n{"question":"same","answer":"#### 2"}\n'),
    ],
)
def test_strict_jsonl_loader_fails_closed_on_empty_malformed_or_duplicate_data(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(GSM8KDataError):
        load_gsm8k_jsonl(path)


def test_strict_jsonl_loader_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.jsonl"
    path.write_bytes(b'{"question":"\xff","answer":"#### 1"}\n')

    with pytest.raises(GSM8KDataError, match="strict UTF-8"):
        load_gsm8k_jsonl(path)


def test_template_normalization_groups_numeric_variants_without_crossing_groups() -> None:
    first = "  MIA has 3 apples and buys 2 more.  "
    second = "Mia has 30 apples and buys 20 more."

    assert normalize_gsm8k_template(first) == ("mia has <number> apples and buys <number> more.")
    assert gsm8k_template_family_sha256(first) == gsm8k_template_family_sha256(second)
    assert gsm8k_template_family_sha256(first) != gsm8k_template_family_sha256(
        "Mia sells 3 apples."
    )
    with pytest.raises(ValueError):
        normalize_gsm8k_template("   ")


def test_adapter_defaults_to_byte_exact_pinned_sources(gsm8k_files: tuple[Path, Path]) -> None:
    with pytest.raises(GSM8KDataError, match="pinned GSM8K source verification failed"):
        GSM8KBenchmarkAdapter(*gsm8k_files)


def test_pinned_default_split_invariants_are_explicit() -> None:
    assert dict(GSM8K_PINNED_DEFAULT_PARTITION_COUNTS) == {
        ProtocolPartition.EXPLORATION: 5_978,
        ProtocolPartition.GATE: 1_495,
        ProtocolPartition.SEALED: 1_319,
    }
    assert len(GSM8K_PINNED_DEFAULT_SPLIT_FINGERPRINT) == 64
    assert len(GSM8K_PINNED_DEFAULT_ADAPTER_FINGERPRINT) == 64


def test_adapter_exposes_only_opaque_id_and_question_to_candidate(
    adapter: GSM8KBenchmarkAdapter,
) -> None:
    rosters = {partition: adapter.task_roster(partition) for partition in ProtocolPartition}
    all_ids = tuple(task_id for roster in rosters.values() for task_id in roster)

    assert all(rosters.values())
    assert len(all_ids) == len(set(all_ids))
    assert all(task_id.startswith("gsm8k-") and len(task_id) == 70 for task_id in all_ids)
    task = adapter.load_task(rosters[ProtocolPartition.EXPLORATION][0])
    assert set(type(task).model_fields) == {"task_id", "question"}
    assert not hasattr(task, "answer")
    assert not hasattr(task, "source_index")
    assert not hasattr(task, "partition")
    assert isinstance(adapter, BenchmarkAdapter)


def test_train_template_families_never_cross_exploration_and_gate(
    adapter: GSM8KBenchmarkAdapter,
) -> None:
    manifest = adapter.split_manifest
    exploration = manifest.partition(ProtocolPartition.EXPLORATION)
    gate = manifest.partition(ProtocolPartition.GATE)
    sealed = manifest.partition(ProtocolPartition.SEALED)
    exploration_families = {task.template_family_sha256 for task in exploration.tasks}
    gate_families = {task.template_family_sha256 for task in gate.tasks}

    assert exploration_families.isdisjoint(gate_families)
    assert len(exploration.tasks) + len(gate.tasks) == len(TRAIN_ROWS)
    assert len(sealed.tasks) == len(TEST_ROWS)
    numeric_family = gsm8k_template_family_sha256(TRAIN_ROWS[0]["question"])
    numeric_memberships = [
        numeric_family in exploration_families,
        numeric_family in gate_families,
    ]
    assert sum(numeric_memberships) == 1


def test_official_test_questions_are_sealed_only(adapter: GSM8KBenchmarkAdapter) -> None:
    questions_by_partition = {
        partition: {
            adapter.load_task(task_id).question for task_id in adapter.task_roster(partition)
        }
        for partition in ProtocolPartition
    }

    assert questions_by_partition[ProtocolPartition.SEALED] == {
        row["question"] for row in TEST_ROWS
    }
    train_questions = {row["question"] for row in TRAIN_ROWS}
    assert (
        questions_by_partition[ProtocolPartition.EXPLORATION].union(
            questions_by_partition[ProtocolPartition.GATE]
        )
        == train_questions
    )
    assert questions_by_partition[ProtocolPartition.SEALED].isdisjoint(train_questions)


def test_split_manifest_and_adapter_fingerprints_are_replayable(
    gsm8k_files: tuple[Path, Path],
) -> None:
    first = GSM8KBenchmarkAdapter(*gsm8k_files, verify_pinned=False)
    second = GSM8KBenchmarkAdapter(*gsm8k_files, verify_pinned=False)

    assert first.split_manifest == second.split_manifest
    assert first.split_manifest.fingerprint == second.split_manifest.fingerprint
    assert first.fingerprint == second.fingerprint
    assert first.split_manifest.config == DEFAULT_GSM8K_SPLIT_CONFIG
    assert len({source.sha256 for source in first.split_manifest.sources}) == 2


def test_split_seed_and_config_are_frozen_into_fingerprint(
    gsm8k_files: tuple[Path, Path],
) -> None:
    alternate_config = GSM8KSplitConfig(seed=99, gate_fraction=0.4)
    default = GSM8KBenchmarkAdapter(*gsm8k_files, verify_pinned=False)
    alternate = GSM8KBenchmarkAdapter(
        *gsm8k_files,
        split_config=alternate_config,
        verify_pinned=False,
    )

    assert default.split_manifest.config.seed != alternate.split_manifest.config.seed
    assert default.split_manifest.fingerprint != alternate.split_manifest.fingerprint
    assert default.fingerprint != alternate.fingerprint


def test_split_requires_at_least_two_train_template_families(tmp_path: Path) -> None:
    train_path = write_jsonl(tmp_path / "train.jsonl", TRAIN_ROWS[:2])
    test_path = write_jsonl(tmp_path / "test.jsonl", TEST_ROWS)

    with pytest.raises(GSM8KDataError, match="at least two distinct template families"):
        GSM8KBenchmarkAdapter(train_path, test_path, verify_pinned=False)


def test_split_manifest_rejects_roster_and_family_overlap(adapter: GSM8KBenchmarkAdapter) -> None:
    manifest = adapter.split_manifest
    exploration = manifest.partition(ProtocolPartition.EXPLORATION)
    gate = manifest.partition(ProtocolPartition.GATE)
    sealed = manifest.partition(ProtocolPartition.SEALED)
    duplicate_task = GSM8KTaskManifestEntry(
        task_id=exploration.tasks[0].task_id,
        template_family_sha256=exploration.tasks[0].template_family_sha256,
        source_role="test",
    )
    bad_sealed = GSM8KPartitionManifest(
        partition=ProtocolPartition.SEALED,
        tasks=(duplicate_task, *sealed.tasks),
    )
    with pytest.raises(ValidationError, match="rosters must be disjoint"):
        GSM8KSplitManifest(
            sources=manifest.sources,
            config=manifest.config,
            partitions=(exploration, gate, bad_sealed),
        )

    crossed_family = GSM8KTaskManifestEntry(
        task_id=gate.tasks[0].task_id,
        template_family_sha256=exploration.tasks[0].template_family_sha256,
        source_role="train",
    )
    bad_gate = GSM8KPartitionManifest(
        partition=ProtocolPartition.GATE,
        tasks=(crossed_family, *gate.tasks[1:]),
    )
    with pytest.raises(ValidationError, match="template family"):
        GSM8KSplitManifest(
            sources=manifest.sources,
            config=manifest.config,
            partitions=(exploration, bad_gate, sealed),
        )

    crossed_sealed_family = GSM8KTaskManifestEntry(
        task_id=sealed.tasks[0].task_id,
        template_family_sha256=exploration.tasks[0].template_family_sha256,
        source_role="test",
    )
    bad_sealed_family = GSM8KPartitionManifest(
        partition=ProtocolPartition.SEALED,
        tasks=(crossed_sealed_family, *sealed.tasks[1:]),
    )
    with pytest.raises(ValidationError, match="template family"):
        GSM8KSplitManifest(
            sources=manifest.sources,
            config=manifest.config,
            partitions=(exploration, gate, bad_sealed_family),
        )

    test_in_gate = GSM8KTaskManifestEntry(
        task_id=gate.tasks[0].task_id,
        template_family_sha256=gate.tasks[0].template_family_sha256,
        source_role="test",
    )
    bad_gate_source = GSM8KPartitionManifest(
        partition=ProtocolPartition.GATE,
        tasks=(test_in_gate, *gate.tasks[1:]),
    )
    with pytest.raises(ValidationError, match="must map only"):
        GSM8KSplitManifest(
            sources=manifest.sources,
            config=manifest.config,
            partitions=(exploration, bad_gate_source, sealed),
        )


class RunnerStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ScoreFreeExecution:
    task_id: str
    seed: int
    harness_id: str
    output_text: str | None
    status: RunnerStatus = RunnerStatus.COMPLETED
    tokens: int = 42
    latency_ms: float = 12.5
    tool_calls: int = 0
    cost_usd: float | None = None
    execution_fingerprint: str = "f" * 64


def execution_for(task: GSM8KTask, output: str | None) -> ScoreFreeExecution:
    return ScoreFreeExecution(
        task_id=task.task_id,
        seed=7,
        harness_id="candidate",
        output_text=output,
    )


def grade(
    adapter: GSM8KBenchmarkAdapter,
    task: GSM8KTask,
    execution: ScoreFreeExecution,
):
    return adapter.grade(
        task,
        execution,
        harness_id="candidate",
        seed=7,
        execution_fingerprint="f" * 64,
    )


def test_trusted_grader_uses_official_semantics_and_copies_runner_usage(
    adapter: GSM8KBenchmarkAdapter,
) -> None:
    task_id = next(
        task_id
        for task_id in adapter.task_roster(ProtocolPartition.SEALED)
        if "earns" in adapter.load_task(task_id).question
    )
    task = adapter.load_task(task_id)

    correct = grade(adapter, task, execution_for(task, "first marker #### 950 then #### 1"))
    wrong = grade(adapter, task, execution_for(task, "no official marker: 950"))

    assert correct.score == 1.0
    assert correct.status is TrialStatus.COMPLETED
    assert correct.tokens == 42
    assert correct.latency_ms == 12.5
    assert correct.tool_calls == 0
    assert correct.slice_tags == ("benchmark:gsm8k",)
    assert wrong.score == 0.0


def test_adapter_grades_the_verified_snapshot_when_source_path_is_replaced(
    gsm8k_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path, test_path = gsm8k_files
    original_train = train_path.read_bytes()
    original_sha256 = hashlib.sha256(original_train).hexdigest()
    original_reader = gsm8k_module.read_dataset_snapshot
    tampered_rows = tuple(
        {**row, "answer": "tampered after snapshot verification\n#### 0"} for row in TRAIN_ROWS
    )

    def read_then_replace(path, *, artifact=None, chunk_size=1024 * 1024):
        snapshot = original_reader(path, artifact=artifact, chunk_size=chunk_size)
        if Path(path) == train_path:
            write_jsonl(train_path, tampered_rows)
        return snapshot

    monkeypatch.setattr(gsm8k_module, "read_dataset_snapshot", read_then_replace)
    adapter = GSM8KBenchmarkAdapter(train_path, test_path, verify_pinned=False)
    task_ids = (
        *adapter.task_roster(ProtocolPartition.EXPLORATION),
        *adapter.task_roster(ProtocolPartition.GATE),
    )
    task = next(
        adapter.load_task(task_id)
        for task_id in task_ids
        if adapter.load_task(task_id).question == TRAIN_ROWS[0]["question"]
    )
    train_source = next(
        source for source in adapter.split_manifest.sources if source.role == "train"
    )

    assert train_path.read_bytes() != original_train
    assert train_source.sha256 == original_sha256
    assert grade(adapter, task, execution_for(task, "#### 5")).score == 1.0
    assert grade(adapter, task, execution_for(task, "#### 0")).score == 0.0


def test_failed_runner_execution_is_unscored_infra_evidence(
    adapter: GSM8KBenchmarkAdapter,
) -> None:
    task = adapter.load_task(adapter.task_roster(ProtocolPartition.GATE)[0])
    failed = replace(
        execution_for(task, None),
        status=RunnerStatus.FAILED,
        tokens=3,
        latency_ms=1.0,
    )

    observation = grade(adapter, task, failed)

    assert observation.status is TrialStatus.INFRA_ERROR
    assert observation.score is None
    assert observation.tokens == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "other"),
        ("seed", 8),
        ("harness_id", "other"),
        ("execution_fingerprint", "e" * 64),
    ],
)
def test_grader_rejects_execution_provenance_substitution(
    adapter: GSM8KBenchmarkAdapter,
    field: str,
    value: object,
) -> None:
    task = adapter.load_task(adapter.task_roster(ProtocolPartition.GATE)[0])
    execution = replace(execution_for(task, "#### 0"), **{field: value})

    with pytest.raises(GSM8KGradingError, match="provenance"):
        grade(adapter, task, execution)


def test_grader_rejects_task_substitution_and_invalid_telemetry(
    adapter: GSM8KBenchmarkAdapter,
) -> None:
    task = adapter.load_task(adapter.task_roster(ProtocolPartition.GATE)[0])
    altered_task = GSM8KTask(task_id=task.task_id, question="Altered question")
    with pytest.raises(GSM8KGradingError, match="does not match"):
        grade(adapter, altered_task, execution_for(task, "#### 0"))

    with pytest.raises(GSM8KGradingError, match="telemetry"):
        grade(adapter, task, replace(execution_for(task, "#### 0"), tokens=-1))
    with pytest.raises(GSM8KGradingError, match="must contain output"):
        grade(adapter, task, execution_for(task, None))
    with pytest.raises(GSM8KGradingError, match="required score-free telemetry"):
        adapter.grade(
            task,
            object(),
            harness_id="candidate",
            seed=7,
            execution_fingerprint="f" * 64,
        )


def test_adapter_rejects_unknown_refs_partitions_and_unregistered_tasks(
    adapter: GSM8KBenchmarkAdapter,
) -> None:
    with pytest.raises(GSM8KDataError, match="unknown GSM8K task"):
        adapter.load_task("gsm8k-" + "0" * 64)
    with pytest.raises(GSM8KDataError, match="non-empty"):
        adapter.load_task("")
    with pytest.raises(GSM8KDataError, match="unknown GSM8K protocol partition"):
        adapter.task_roster("gate")  # type: ignore[arg-type]

    unknown = GSM8KTask(task_id="gsm8k-" + "0" * 64, question="Unknown")
    with pytest.raises(GSM8KGradingError, match="outside the frozen"):
        grade(adapter, unknown, execution_for(unknown, "#### 0"))
