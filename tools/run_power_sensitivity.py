"""Run a non-authoritative power sensitivity proxy from explicit assumptions.

This command has no formal design-freeze mode.  An optional caller-declared
pilot manifest is canonicalized and digest-bound, but its referenced artifacts
and empirical provenance are not verified and cannot make the proxy design-ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.experiments.power_analysis import (
    PowerSensitivityConfig,
    run_power_sensitivity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON sensitivity assumptions")
    parser.add_argument("output", type=Path, help="new path for the canonical proxy report")
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        help="separate canonical declared-pilot manifest bound by config digest",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output path explicitly",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = PowerSensitivityConfig.model_validate_json(args.config.read_bytes(), strict=True)
        manifest_payload = None
        if config.declared_pilot_manifest_expectation is None:
            if args.pilot_manifest is not None:
                raise ValueError(
                    "--pilot-manifest is forbidden unless config binds a manifest expectation"
                )
        else:
            if args.pilot_manifest is None:
                raise ValueError(
                    "config binds a pilot manifest expectation; --pilot-manifest is required"
                )
            manifest_payload = args.pilot_manifest.read_bytes()
        report = run_power_sensitivity(
            config,
            declared_pilot_manifest_payload=manifest_payload,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if args.overwrite else "xb"
        with args.output.open(mode) as stream:
            stream.write(canonical_json_bytes(report))
    except (OSError, ValidationError, ValueError) as error:
        parser.error(str(error))

    summary = {
        "analysis_status": report.analysis_status,
        "config_sha256": report.config_sha256,
        "formal_design_ready": report.formal_design_ready,
        "formal_required_search_seeds_per_cell": (report.formal_required_search_seeds_per_cell),
        "output": str(args.output.resolve()),
        "declared_pilot_manifest_digest_bound": report.declared_pilot_manifest_digest_bound,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
