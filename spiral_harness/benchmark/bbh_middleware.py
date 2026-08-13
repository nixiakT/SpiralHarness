"""Declarative, score-blind output normalization for BBH choice answers."""

from __future__ import annotations

import re
from typing import Literal

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel

BBH_OUTPUT_MIDDLEWARE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bbh-output-contract-middleware.v1+json"
)
_FINAL_DECLARATION_RE = re.compile(
    r"(?i)(?:^|\n)\s*(?:final\s+answer|answer|only\s+\([A-G]\)\s+is\s+correct)"
    r"\s*(?:is|:)?\s*(\([A-G]\))(?:[^\n]*)\s*$"
)
_STANDALONE_RE = re.compile(r"(?m)^\s*(\([A-G]\))\s*$")


class BBHOutputContractMiddleware(ImmutableModel):
    """Normalize only an explicit final answer declaration, never infer an option."""

    schema_version: Literal["1"] = "1"
    rule: Literal["explicit-final-choice-to-standalone-line-v1"] = (
        "explicit-final-choice-to-standalone-line-v1"
    )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    def apply(self, output: str) -> str:
        if type(output) is not str:
            raise TypeError("BBH middleware input must be a string")
        standalone = _STANDALONE_RE.findall(output)
        if standalone:
            return output.rstrip() + "\n"
        match = _FINAL_DECLARATION_RE.search(output)
        if match is None:
            return output
        return output.rstrip() + "\n\n" + match.group(1).upper() + "\n"


__all__ = ["BBH_OUTPUT_MIDDLEWARE_MEDIA_TYPE", "BBHOutputContractMiddleware"]
