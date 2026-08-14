"""Frozen proposal grammar and feedback projections for the development study."""

from __future__ import annotations

import re

from spiral_harness.core.canonical import canonical_json
from spiral_harness.experiments.development_four_arm_contracts import (
    FullDisclosure,
    ScoreDisclosure,
)
from spiral_harness.skills.package import RESERVED_SKILL_CONTEXT_DELIMITERS

DEVELOPMENT_SEED_PROMPT = """You solve each task carefully and independently.
For arithmetic word problems, show concise reasoning and end with exactly one line `#### <number>`.
For multiple-choice logic problems, show concise reasoning and end with the selected option alone,
such as `(A)`. Never invent missing facts and verify the requested final format."""

_PROPOSAL_GRAMMAR = (
    "Return exactly one non-empty <HARNESS_PROMPT>...</HARNESS_PROMPT> block and nothing else. "
    "The inner text becomes the complete solver system prompt. Do not quote the tags inside it."
)

SCORE_PROPOSER_SYSTEM_PROMPT = (
    "You improve a solver harness using aggregate fit scores only. "
    "You receive no item outputs, correctness labels, trajectories, gold answers, or holdout data. "
    "Propose one general prompt that could improve both benchmark types. " + _PROPOSAL_GRAMMAR
)

FULL_PROPOSER_SYSTEM_PROMPT = (
    "You improve a solver harness using authorized fit-task questions, solver outputs, and binary "
    "correctness evidence. Infer reusable failure mechanisms; do not memorize task answers or "
    "request holdout information. Propose one general prompt that could improve both benchmark "
    "types. " + _PROPOSAL_GRAMMAR
)

_PROMPT_BLOCK_RE = re.compile(
    r"\A\s*<HARNESS_PROMPT>\s*(.*?)\s*</HARNESS_PROMPT>\s*\Z",
    re.DOTALL,
)
_MAX_CANDIDATE_PROMPT_CHARS = 8_000


def build_score_proposal_input(
    disclosure: ScoreDisclosure,
) -> str:
    """Render SCORE's aggregate-only projection without adding item evidence."""

    checked = ScoreDisclosure.model_validate(disclosure, strict=True)
    return (
        "<PARENT_SYSTEM_PROMPT>\n"
        + DEVELOPMENT_SEED_PROMPT
        + "\n</PARENT_SYSTEM_PROMPT>\n\n"
        + "<AGGREGATE_FIT_RESULTS>\n"
        + _delimiter_safe_canonical_json(checked)
        + "\n</AGGREGATE_FIT_RESULTS>\n\n"
        + _PROPOSAL_GRAMMAR
    )


def build_full_proposal_input(
    disclosure: FullDisclosure,
) -> str:
    """Render FULL's authorized item-level fit evidence, never holdout or gold."""

    checked = FullDisclosure.model_validate(disclosure, strict=True)
    return (
        "<PARENT_SYSTEM_PROMPT>\n"
        + DEVELOPMENT_SEED_PROMPT
        + "\n</PARENT_SYSTEM_PROMPT>\n\n"
        + "<FIT_ITEM_EVIDENCE>\n"
        + _delimiter_safe_canonical_json(checked)
        + "\n</FIT_ITEM_EVIDENCE>\n\n"
        + _PROPOSAL_GRAMMAR
    )


def parse_candidate_prompt(model_output: str | None) -> str | None:
    """Parse one bounded candidate; invalid output declines without retry or repair."""

    if not isinstance(model_output, str):
        return None
    match = _PROMPT_BLOCK_RE.fullmatch(model_output)
    if match is None:
        return None
    candidate = match.group(1).strip()
    if not candidate or len(candidate) > _MAX_CANDIDATE_PROMPT_CHARS:
        return None
    if "<HARNESS_PROMPT>" in candidate or "</HARNESS_PROMPT>" in candidate:
        return None
    if any(delimiter in candidate for delimiter in RESERVED_SKILL_CONTEXT_DELIMITERS):
        return None
    if "\x00" in candidate:
        return None
    return candidate


def _delimiter_safe_canonical_json(value: object) -> str:
    """Keep evidence as JSON while preventing data from spelling control delimiters."""

    return (
        canonical_json(value)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


__all__ = [
    "DEVELOPMENT_SEED_PROMPT",
    "FULL_PROPOSER_SYSTEM_PROMPT",
    "SCORE_PROPOSER_SYSTEM_PROMPT",
    "build_full_proposal_input",
    "build_score_proposal_input",
    "parse_candidate_prompt",
]
