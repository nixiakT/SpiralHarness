from __future__ import annotations

from pathlib import Path

from tools.report_junit_failures import annotations_from_junit


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
