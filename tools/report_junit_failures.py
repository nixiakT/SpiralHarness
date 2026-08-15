"""Publish pytest JUnit failures as GitHub Actions annotations.

GitHub no longer exposes complete Actions logs to anonymous readers.  Keeping
the failure name and traceback in check-run annotations makes a public CI
failure diagnosable without granting repository credentials.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_ANNOTATION_DETAIL_LIMIT = 3_500


def _escape_message(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_message(value).replace(":", "%3A").replace(",", "%2C")


def annotations_from_junit(path: Path) -> tuple[str, ...]:
    """Return deterministic workflow annotations for every failed test case."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        detail = _escape_message(f"pytest failed and JUnit could not be read: {error}")
        return (f"::error title=pytest JUnit unavailable::{detail}",)

    annotations: list[str] = []
    for case in root.iter("testcase"):
        failures = (*case.findall("failure"), *case.findall("error"))
        for failure in failures:
            classname = case.get("classname", "pytest")
            name = case.get("name", "unknown test")
            title = _escape_property(f"{classname}::{name}")
            raw_detail = (failure.text or failure.get("message") or "pytest failure").strip()
            # GitHub truncates annotation messages to roughly 4 KiB from the
            # right, so sending an 8 KiB traceback discarded the assertion and
            # exception at its tail.  Escape first, then retain the diagnostic
            # tail under a conservative wire-size ceiling.
            escaped_detail = _escape_message(raw_detail)
            detail = escaped_detail[-_ANNOTATION_DETAIL_LIMIT:]
            annotations.append(f"::error title={title}::{detail}")
    if not annotations:
        annotations.append(
            "::error title=pytest failed::The test command failed without a JUnit failure node"
        )
    return tuple(annotations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit", type=Path)
    args = parser.parse_args(argv)
    for annotation in annotations_from_junit(args.junit):
        print(annotation)
    return 1


if __name__ == "__main__":
    sys.exit(main())
