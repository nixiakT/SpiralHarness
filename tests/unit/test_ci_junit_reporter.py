from __future__ import annotations

import ast
from pathlib import Path

from tools.report_junit_failures import annotations_from_junit

ROOT = Path(__file__).resolve().parents[2]


def test_junit_reporter_emits_test_identity_and_escaped_traceback(tmp_path: Path) -> None:
    report = tmp_path / "pytest.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite failures="1">
  <testcase classname="tests.test_demo" name="test_bad">
    <failure message="failed">left != right\n100%</failure>
  </testcase>
</testsuite></testsuites>
""",
        encoding="utf-8",
    )

    assert annotations_from_junit(report) == (
        "::error title=tests.test_demo%3A%3Atest_bad::left != right%0A100%25",
    )


def test_junit_reporter_fails_closed_when_report_is_missing(tmp_path: Path) -> None:
    annotations = annotations_from_junit(tmp_path / "missing.xml")

    assert len(annotations) == 1
    assert annotations[0].startswith("::error title=pytest JUnit unavailable::")


def test_junit_reporter_preserves_the_failure_tail_within_github_limit(
    tmp_path: Path,
) -> None:
    report = tmp_path / "pytest.xml"
    oversized_prefix = "source line\n" * 1_000
    report.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite failures="1">'
        '<testcase classname="tests.test_demo" name="test_long">'
        f"<failure>{oversized_prefix}FINAL ASSERTION: expected 1, got 0</failure>"
        "</testcase></testsuite></testsuites>",
        encoding="utf-8",
    )

    annotation = annotations_from_junit(report)[0]

    assert len(annotation) < 4_096
    assert "FINAL ASSERTION: expected 1, got 0" in annotation
    assert oversized_prefix not in annotation


def test_workflow_runs_reporter_only_after_a_real_pytest_failure() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "continue-on-error: true" not in workflow
    assert "if: failure() && steps.pytest.outcome == 'failure'" in workflow
    assert workflow.index("id: pytest") < workflow.index(
        "if: failure() && steps.pytest.outcome == 'failure'"
    )


def test_repository_python_text_io_declares_utf8_explicitly() -> None:
    missing_encoding: list[str] = []
    for relative_root in ("spiral_harness", "tests", "tools", "benchmarks"):
        for path in sorted((ROOT / relative_root).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in {"read_text", "write_text"}:
                    continue
                if not any(keyword.arg == "encoding" for keyword in node.keywords):
                    relative = path.relative_to(ROOT)
                    missing_encoding.append(f"{relative}:{node.lineno}:{node.func.attr}")

    assert missing_encoding == []
