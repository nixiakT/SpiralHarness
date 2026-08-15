#!/usr/bin/env python3
"""Minimal-environment smoke for one public BFCL evaluator function."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

EXPECTED_VERSION = "2026.3.23"
ALLOWED_ENVIRONMENT_NAMES = frozenset({"LC_ALL", "PATH", "PYTHONDONTWRITEBYTECODE", "TZ"})
AUDITED_SOCKET_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    return parser


def main() -> int:
    if not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode:
        raise RuntimeError("smoke requires Python flags -I -S -B")
    unexpected_environment = sorted(set(os.environ) - ALLOWED_ENVIRONMENT_NAMES)
    if unexpected_environment:
        raise RuntimeError(f"smoke environment is not minimal: {unexpected_environment}")

    args = _parser().parse_args()
    if not args.venv.is_absolute() or args.venv.is_symlink():
        raise RuntimeError("smoke virtualenv must be an absolute non-symlink path")
    venv = args.venv.resolve(strict=True)
    site_packages = venv / "lib/python3.12/site-packages"
    if site_packages.is_symlink() or not site_packages.is_dir():
        raise RuntimeError("smoke site-packages is missing or a symlink")

    # ``-S`` prevents site initialization and .pth execution.  Add only the
    # previously verified site-packages directory after checking its location.
    sys.path.insert(0, str(site_packages))
    if importlib.metadata.version("bfcl-eval") != EXPECTED_VERSION:
        raise RuntimeError("the installed bfcl-eval release is not pinned")

    audited_socket_events_observed: list[str] = []

    def observe_socket_events(event: str, _args: tuple[object, ...]) -> None:
        if event in AUDITED_SOCKET_EVENTS:
            audited_socket_events_observed.append(event)
            # This is a narrow CPython-level tripwire, not syscall isolation.
            raise RuntimeError(f"observed socket audit event during evaluator smoke: {event}")

    sys.addaudithook(observe_socket_events)
    from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker

    accepted = agentic_checker(
        "The normalized answer is April 1, 2024.",
        ["April 1 2024"],
    )
    rejected = agentic_checker("No matching value is present.", ["expected-value"])
    if accepted.get("valid") is not True or rejected.get("valid") is not False:
        raise RuntimeError("public BFCL agentic checker failed the fixed smoke cases")
    print(
        json.dumps(
            {
                "kind": "bfcl-v4-public-evaluator-import-smoke",
                "bfcl_eval": EXPECTED_VERSION,
                "accepted_case": True,
                "rejected_case": True,
                "provider_calls_requested": 0,
                "audited_socket_events_observed": audited_socket_events_observed,
                "dependency_environment_attested": False,
                "network_isolation_attested": False,
                "syscall_sandbox_attested": False,
                "reportable_result": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
