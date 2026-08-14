"""Auditable reconstructions of paper-described Meta-Harness classifiers.

The Meta-Harness release does not contain the final discovered classifier source files.
This module therefore implements the algorithms described in Appendix B.1 of the paper,
without claiming source-level reproduction.  It deliberately has no dependency on the
upstream runner so the prompt construction and retrieval policy can be unit tested here and
adapted to the upstream ``MemorySystem`` interface during a controlled benchmark.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

META_HARNESS_PAPER_ARXIV = "2603.28052"
META_HARNESS_REVISION = "44b9942127847f7421db70d8c7e48407f09a3c70"
PAPER_HARNESS_NAME = "Label-Primed Query (paper algorithm reconstruction)"

_LATIN_OR_NUMBER = re.compile(r"[a-z0-9]+")
_HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_MULTI_LABEL_SEPARATOR = re.compile("[;\uff1b]")
_DIAGNOSIS_TAG = re.compile(r"\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]", re.I | re.S)
_CHARGE_TAG = re.compile(r"\[罪名\](.*?)(?:<eoa(?:>|\])|$)", re.S)


@dataclass(frozen=True, slots=True)
class LabeledMemoryExample:
    """One training example visible to an offline memory harness."""

    input: str
    target: str
    raw_question: str

    @classmethod
    def from_batch_result(cls, result: dict[str, Any]) -> LabeledMemoryExample:
        return cls(
            input=str(result["input"]),
            target=str(result["ground_truth"]),
            raw_question=str(result.get("raw_question", result["input"])),
        )


def atomic_labels(target: str) -> tuple[str, ...]:
    """Return atomic labels while preserving commas inside Chinese charge names."""

    labels = tuple(part.strip() for part in _MULTI_LABEL_SEPARATOR.split(target) if part.strip())
    return labels or (target.strip(),)


def clean_classification_answer(answer: str) -> str:
    """Remove benchmark wrappers, including a common malformed closing bracket.

    This is output-contract normalization rather than target correction: the function never
    invents or maps a label, and therefore remains independent of dataset answers.
    """

    text = answer.strip()
    diagnosis = _DIAGNOSIS_TAG.search(text)
    if diagnosis:
        return diagnosis.group(1).strip()
    charge = _CHARGE_TAG.search(text)
    if charge:
        return charge.group(1).strip()
    return text


def route_by_validation_score_band(
    *,
    local_answer: str,
    label_primed_answer: str,
    ranked_candidates: tuple[tuple[str, float], ...],
) -> str:
    """Apply the validation-selected low-label-space routing rule.

    Scores are relative log likelihoods with the best candidate at zero. The score bands
    are frozen hyperparameters selected on validation traces; this function never sees an
    evaluation target.
    """

    local = clean_classification_answer(local_answer)
    paper = clean_classification_answer(label_primed_answer)
    if len(ranked_candidates) < 2:
        return local
    best = ranked_candidates[0][0]
    margin = ranked_candidates[0][1] - ranked_candidates[1][1]
    use_statistical_candidate = paper == best or margin < 1.0 or 3.0 < margin < 8.0
    return best if use_statistical_candidate else local


def route_by_taxonomy_specificity(
    *,
    model_only_answer: str,
    draft_verified_answer: str,
    label_primed_answer: str,
    local_evidence_answer: str | None = None,
) -> str:
    """Reconcile sparse-taxonomy candidates without a dataset-specific label map."""

    model_only = clean_classification_answer(model_only_answer)
    draft = clean_classification_answer(draft_verified_answer)
    paper = clean_classification_answer(label_primed_answer)
    paper_labels = set(atomic_labels(paper))
    draft_labels = set(atomic_labels(draft))
    choose_paper = paper_labels < draft_labels
    paper_only = paper_labels - draft_labels
    draft_only = draft_labels - paper_labels
    if len(paper_only) == len(draft_only) == 1:
        paper_label = next(iter(paper_only))
        draft_label = next(iter(draft_only))
        if (
            len(paper_label) > len(draft_label)
            and SequenceMatcher(None, paper_label, draft_label).ratio() > 0.5
        ) or (paper_label in draft_label and model_only != draft):
            choose_paper = True
    selected = paper if choose_paper else draft
    if local_evidence_answer is not None:
        local = clean_classification_answer(local_evidence_answer)
        model_labels = set(atomic_labels(model_only))
        local_labels = set(atomic_labels(local))
        # A validation-discovered decomposition: when the broad-context candidate is exactly
        # the union of the model prior and an independent compact-memory candidate, and draft
        # verification retains the prior plus a conflicting alternative, select the compact
        # candidate rather than unioning labels.
        if (
            paper_labels == model_labels | local_labels
            and model_labels.isdisjoint(local_labels)
            and model_labels < draft_labels
        ):
            selected = local
    return selected


def route_by_majority(candidates: Iterable[str], *, fallback: str) -> str:
    """Choose a normalized plurality label with a deterministic fallback for ties."""

    normalized = tuple(clean_classification_answer(candidate) for candidate in candidates)
    if not normalized:
        return clean_classification_answer(fallback)
    counts = Counter(normalized)
    best_count = max(counts.values())
    winners = {candidate for candidate, count in counts.items() if count == best_count}
    normalized_fallback = clean_classification_answer(fallback)
    return normalized_fallback if normalized_fallback in winners else sorted(winners)[0]


def unicode_terms(text: str) -> Counter[str]:
    """Tokenize English-like text and Han text without language-specific dependencies."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    terms: Counter[str] = Counter(_LATIN_OR_NUMBER.findall(normalized))
    for run in _HAN_RUN.findall(normalized):
        terms.update(run)
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def classification_terms(text: str) -> Counter[str]:
    """Add short English phrases to the language-neutral retrieval features."""

    terms = unicode_terms(text)
    words = _LATIN_OR_NUMBER.findall(unicodedata.normalize("NFKC", text).lower())
    for size in (2, 3):
        terms.update(
            " ".join(words[index : index + size]) for index in range(len(words) - size + 1)
        )
    return terms


class UnicodeTfidfIndex:
    """Small deterministic TF-IDF index suitable for benchmark memories."""

    def __init__(self, documents: Iterable[str]) -> None:
        self.documents = tuple(documents)
        term_counts = tuple(unicode_terms(document) for document in self.documents)
        document_frequency = Counter(term for counts in term_counts for term in counts)
        count = len(self.documents)
        self.idf = {
            term: math.log((count + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = tuple(self._vector(counts) for counts in term_counts)
        self.norms = tuple(self._norm(vector) for vector in self.vectors)

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        return {
            term: (1 + math.log(frequency)) * self.idf.get(term, 1.0)
            for term, frequency in counts.items()
        }

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    @staticmethod
    def _cosine(
        left: dict[str, float],
        left_norm: float,
        right: dict[str, float],
        right_norm: float,
    ) -> float:
        if not left_norm or not right_norm:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        dot = sum(value * right.get(term, 0.0) for term, value in left.items())
        return dot / (left_norm * right_norm)

    def query_scores(self, query: str) -> tuple[float, ...]:
        vector = self._vector(unicode_terms(query))
        norm = self._norm(vector)
        return tuple(
            self._cosine(vector, norm, document, document_norm)
            for document, document_norm in zip(self.vectors, self.norms, strict=True)
        )

    def document_similarity(self, left: int, right: int) -> float:
        return self._cosine(
            self.vectors[left], self.norms[left], self.vectors[right], self.norms[right]
        )

    def rank(self, query: str, candidates: Iterable[int] | None = None) -> tuple[int, ...]:
        scores = self.query_scores(query)
        indices = range(len(self.documents)) if candidates is None else candidates
        return tuple(sorted(indices, key=lambda index: (-scores[index], index)))


class LabelPrimedQueryMemory:
    """Reconstruct the paper's strongest text-classification harness.

    Appendix B.1 specifies four observable mechanisms: a valid-label primer, one
    query-relevant coverage example per label, query-local contrastive pairs, and TF-IDF
    retrieval.  Details absent from the paper (tokenization, pair count, wording, and tie
    breaking) are deterministic choices in this implementation and are disclosed in state.
    """

    def __init__(
        self,
        *,
        context_char_budget: int = 100_000,
        contrastive_pairs: int = 8,
        coverage_per_label: int = 1,
    ) -> None:
        if context_char_budget <= 0 or contrastive_pairs < 0 or coverage_per_label <= 0:
            raise ValueError("invalid label-primed-query configuration")
        self.context_char_budget = context_char_budget
        self.contrastive_pairs = contrastive_pairs
        self.coverage_per_label = coverage_per_label
        self.examples: list[LabeledMemoryExample] = []
        self._index: UnicodeTfidfIndex | None = None

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        self.examples.extend(
            LabeledMemoryExample.from_batch_result(result) for result in batch_results
        )
        self._index = None

    def _ensure_index(self) -> UnicodeTfidfIndex:
        if self._index is None:
            self._index = UnicodeTfidfIndex(example.raw_question for example in self.examples)
        return self._index

    def labels(self) -> tuple[str, ...]:
        return tuple(
            sorted({label for example in self.examples for label in atomic_labels(example.target)})
        )

    def _coverage(self, query: str) -> tuple[tuple[str, int, float], ...]:
        index = self._ensure_index()
        scores = index.query_scores(query)
        by_label: dict[str, list[int]] = defaultdict(list)
        for example_index, example in enumerate(self.examples):
            for label in atomic_labels(example.target):
                by_label[label].append(example_index)
        selected = []
        for label in sorted(by_label):
            ranked = sorted(by_label[label], key=lambda item: (-scores[item], item))
            selected.extend(
                (label, example_index, scores[example_index])
                for example_index in ranked[: self.coverage_per_label]
            )
        return tuple(selected)

    def _contrastive(self, query: str) -> tuple[tuple[int, int], ...]:
        if self.contrastive_pairs == 0:
            return ()
        index = self._ensure_index()
        query_scores = index.query_scores(query)
        local = sorted(range(len(self.examples)), key=lambda item: (-query_scores[item], item))
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        local_pool = local[: max(24, self.contrastive_pairs * 6)]
        for anchor in local_pool:
            anchor_labels = set(atomic_labels(self.examples[anchor].target))
            alternatives = [
                candidate
                for candidate in local_pool
                if candidate != anchor
                and anchor_labels.isdisjoint(atomic_labels(self.examples[candidate].target))
            ]
            if not alternatives:
                continue
            partner = min(
                alternatives,
                key=lambda item: (
                    -(0.55 * query_scores[item] + 0.45 * index.document_similarity(anchor, item)),
                    item,
                ),
            )
            pair = tuple(sorted((anchor, partner)))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append((anchor, partner))
            if len(pairs) >= self.contrastive_pairs:
                break
        return tuple(pairs)

    @staticmethod
    def _example_text(example: LabeledMemoryExample) -> str:
        return f"Q: {example.raw_question}\nA: {example.target}"

    def build_context(self, query: str) -> str:
        if not self.examples:
            return ""
        labels = self.labels()
        sections = [
            "## Valid label primer\n" + " | ".join(labels),
            "## Query-relevant coverage (one best training example per observed label)",
        ]
        used = sum(len(section) + 2 for section in sections)

        # The paper requires coverage for the full observed label space.  When an explicit
        # budget is smaller than that space, keep the most query-relevant representatives.
        coverage = sorted(self._coverage(query), key=lambda item: (-item[2], item[0]))
        coverage_chunks = []
        for label, index, _score in coverage:
            chunk = f"[Coverage label: {label}]\n{self._example_text(self.examples[index])}"
            if used + len(chunk) + 2 > self.context_char_budget:
                continue
            coverage_chunks.append(chunk)
            used += len(chunk) + 2
        sections.append("\n\n".join(coverage_chunks))

        contrastive_chunks = []
        for pair_number, (left, right) in enumerate(self._contrastive(query), start=1):
            chunk = (
                f"[Local contrastive pair {pair_number}]\n"
                f"{self._example_text(self.examples[left])}\n"
                "VERSUS\n"
                f"{self._example_text(self.examples[right])}"
            )
            if used + len(chunk) + 2 > self.context_char_budget:
                continue
            contrastive_chunks.append(chunk)
            used += len(chunk) + 2
        if contrastive_chunks:
            sections.extend(
                ["## Query-anchored contrastive pairs", "\n\n".join(contrastive_chunks)]
            )
        return "\n\n".join(section for section in sections if section)

    def build_prompt(self, input: str) -> str:
        context = self.build_context(input)
        return f"""Classify the problem using the labeled training memory below.

The valid-label primer defines the observed answer space. Coverage examples expose each
class, while contrastive pairs show nearby examples with different labels. Infer the exact
answer required by the problem; do not blindly copy a neighbor.

{context}

## Problem
{input}

Return valid JSON with exactly these fields:
{{"reasoning": "brief evidence-based reasoning", "final_answer": "exact final answer"}}"""

    def get_state(self) -> str:
        return json.dumps(
            {
                "schema_version": "1",
                "kind": "meta-harness-paper-algorithm-reconstruction",
                "paper": META_HARNESS_PAPER_ARXIV,
                "upstream_revision": META_HARNESS_REVISION,
                "paper_harness": PAPER_HARNESS_NAME,
                "context_char_budget": self.context_char_budget,
                "contrastive_pairs": self.contrastive_pairs,
                "coverage_per_label": self.coverage_per_label,
                "examples": [asdict(example) for example in self.examples],
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def set_state(self, state: str) -> None:
        payload = json.loads(state)
        if payload.get("kind") != "meta-harness-paper-algorithm-reconstruction":
            raise ValueError("not a Meta-Harness paper reconstruction state")
        self.context_char_budget = int(payload["context_char_budget"])
        self.contrastive_pairs = int(payload["contrastive_pairs"])
        self.coverage_per_label = int(payload.get("coverage_per_label", 1))
        self.examples = [LabeledMemoryExample(**example) for example in payload["examples"]]
        self._index = None


class SpiralDraftVerificationMemory:
    """A validation-selectable Spiral candidate that verifies a model-only draft.

    Unlike the paper reconstruction, this candidate keeps the model's zero-context answer
    as an explicit prior and uses a compact local memory only to confirm, revise, and
    canonicalize it.  The policy is domain-independent and permits labels absent from the
    observed training memory, which matters for sparse multi-label datasets.
    """

    def __init__(self, *, neighbor_k: int = 16, confirmer_k: int = 5) -> None:
        if neighbor_k <= 0 or confirmer_k < 0:
            raise ValueError("invalid draft-verification configuration")
        self.neighbor_k = neighbor_k
        self.confirmer_k = confirmer_k
        self.examples: list[LabeledMemoryExample] = []
        self._index: UnicodeTfidfIndex | None = None

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        self.examples.extend(
            LabeledMemoryExample.from_batch_result(result) for result in batch_results
        )
        self._index = None

    def _ensure_index(self) -> UnicodeTfidfIndex:
        if self._index is None:
            self._index = UnicodeTfidfIndex(example.raw_question for example in self.examples)
        return self._index

    def labels(self) -> tuple[str, ...]:
        return tuple(
            sorted({label for example in self.examples for label in atomic_labels(example.target)})
        )

    @staticmethod
    def _example_text(example: LabeledMemoryExample) -> str:
        return f"Q: {example.raw_question}\nA: {example.target}"

    def _evidence_indices(self, input: str, draft: str) -> tuple[int, ...]:
        index = self._ensure_index()
        ranked = index.rank(input)
        draft_labels = set(atomic_labels(clean_classification_answer(draft)))
        confirmers = [
            item
            for item in ranked
            if draft_labels.intersection(atomic_labels(self.examples[item].target))
        ][: self.confirmer_k]
        challengers = [
            item
            for item in ranked
            if draft_labels.isdisjoint(atomic_labels(self.examples[item].target))
        ][: self.neighbor_k]
        ordered = list(ranked[: self.neighbor_k]) + confirmers + challengers
        return tuple(dict.fromkeys(ordered))

    def build_verification_prompt(self, input: str, draft: str) -> str:
        if not self.examples:
            evidence = "(no labeled memory available)"
            labels = "(none observed)"
        else:
            evidence = "\n\n".join(
                self._example_text(self.examples[index])
                for index in self._evidence_indices(input, draft)
            )
            labels = " | ".join(self.labels())
        return f"""Verify a first-pass classification using compact, query-local training evidence.

First-pass draft:
{draft}

Observed atomic label spellings (not necessarily exhaustive):
{labels}

Nearest confirmers and challengers from the training split:
{evidence}

Problem:
{input}

Decision rules:
- Keep the draft when the local evidence does not establish a better answer.
- Revise it when the facts or close counterexamples establish a different label.
- Match observed label spellings when they express the intended class, but labels absent from
  memory are allowed.
- Preserve every independently supported label in a multi-label answer; add no unsupported one.
- Put only the exact task answer in final_answer, following the problem's requested format.

Return valid JSON with exactly these fields:
{{"reasoning": "brief verification", "final_answer": "exact final answer"}}"""

    def get_state(self) -> str:
        return json.dumps(
            {
                "schema_version": "1",
                "kind": "spiral-draft-verification",
                "neighbor_k": self.neighbor_k,
                "confirmer_k": self.confirmer_k,
                "examples": [asdict(example) for example in self.examples],
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def set_state(self, state: str) -> None:
        payload = json.loads(state)
        if payload.get("kind") != "spiral-draft-verification":
            raise ValueError("not a Spiral draft-verification state")
        self.neighbor_k = int(payload["neighbor_k"])
        self.confirmer_k = int(payload["confirmer_k"])
        self.examples = [LabeledMemoryExample(**example) for example in payload["examples"]]
        self._index = None


__all__ = [
    "META_HARNESS_PAPER_ARXIV",
    "META_HARNESS_REVISION",
    "PAPER_HARNESS_NAME",
    "LabelPrimedQueryMemory",
    "LabeledMemoryExample",
    "SpiralDraftVerificationMemory",
    "UnicodeTfidfIndex",
    "atomic_labels",
    "classification_terms",
    "clean_classification_answer",
    "route_by_majority",
    "route_by_taxonomy_specificity",
    "route_by_validation_score_band",
    "unicode_terms",
]
