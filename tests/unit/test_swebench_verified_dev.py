import json

from spiral_harness.benchmark.swebench_verified_dev import (
    SwebenchPatchPlan,
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
