import json
from pathlib import Path

from spiral_harness.benchmark.swebench_verified_dev import (
    SwebenchPatchPlan,
    SwebenchVerifiedTask,
    _collect_excerpts,
    _excerpt_windows,
    extract_patch_plan,
)


def test_extract_patch_plan_reads_plain_json() -> None:
    plan = extract_patch_plan(
        json.dumps(
            {
                "file_path": "src/flask/blueprints.py",
                "before": "self.name = name\n",
                "after": "if not name:\n    raise ValueError('x')\n\nself.name = name\n",
                "explanation": "guard empty blueprint names",
            }
        )
    )
    assert plan == SwebenchPatchPlan(
        file_path="src/flask/blueprints.py",
        before="self.name = name\n",
        after="if not name:\n    raise ValueError('x')\n\nself.name = name\n",
        explanation="guard empty blueprint names",
    )


def test_extract_patch_plan_reads_json_embedded_in_reasoning() -> None:
    text = (
        "thinking...\n"
        '{"file_path":"a.py","before":"old","after":"new","explanation":"ok"}\n'
        "more trailing text"
    )
    plan = extract_patch_plan(text)
    assert plan.file_path == "a.py"
    assert plan.before == "old"
    assert plan.after == "new"


def test_extract_patch_plan_reads_labeled_markdown_blocks() -> None:
    text = (
        "Reasoning...\n"
        "`file_path`: `src/flask/blueprints.py`\n"
        "`before`:\n"
        "```python\n"
        "        self.name = name\n"
        "```\n"
        "`after`:\n"
        "```python\n"
        "        if not name:\n"
        "            raise ValueError(\"empty\")\n"
        "        self.name = name\n"
        "```\n"
    )
    plan = extract_patch_plan(text)
    assert plan.file_path == "src/flask/blueprints.py"
    assert plan.before == "        self.name = name\n"
    assert "raise ValueError" in plan.after


def test_excerpt_windows_cap_span_length() -> None:
    windows = _excerpt_windows([12, 180, 220], total_lines=400)
    assert windows == ((1, 24), (168, 192))


def test_collect_excerpts_keeps_prompt_local(tmp_path: Path) -> None:
    src_dir = tmp_path / "src" / "flask"
    tests_dir = tmp_path / "tests"
    src_dir.mkdir(parents=True)
    tests_dir.mkdir()
    blueprints = "\n".join(
        [f"line {index}" for index in range(1, 20)]
        + [
            "class Blueprint:",
            "    def __init__(self, name):",
            "        self.name = name",
            "        self.import_name = 'demo'",
        ]
        + [f"tail {index}" for index in range(25, 80)]
    )
    app_lines = [f"app line {index}" for index in range(1, 240)]
    app_lines[19] = "Blueprint support exists here"
    app_lines[179] = "Blueprint registration also mentioned here"
    (src_dir / "blueprints.py").write_text(blueprints, encoding="utf-8")
    (src_dir / "app.py").write_text("\n".join(app_lines), encoding="utf-8")
    (tests_dir / "test_blueprints.py").write_text("def test_blueprint_name():\n    pass\n", encoding="utf-8")
    task = SwebenchVerifiedTask(
        repo="pallets/flask",
        instance_id="demo",
        base_commit="deadbeef",
        patch="",
        test_patch="",
        problem_statement="Require a non-empty name for Blueprints",
        hints_text="",
        fail_to_pass=("tests/test_blueprints.py::test_blueprint_name",),
        pass_to_pass=(),
        difficulty="<15 min fix",
    )
    excerpts = _collect_excerpts(tmp_path, task)
    assert excerpts[0]["path"] == "src/flask/blueprints.py"
    assert max(item["end_line"] - item["start_line"] + 1 for item in excerpts) <= 48
