"""Spiral candidates used in controlled native Meta-Harness comparisons."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from spiral_harness.benchmark.meta_harness_paper import (
    LabeledMemoryExample,
    LabelPrimedQueryMemory,
    UnicodeTfidfIndex,
    classification_terms,
)


class SpiralLocalEvidenceMemory:
    """Single-call Spiral candidate combining local cases and learned label evidence."""

    def __init__(
        self,
        *,
        neighbor_k: int = 12,
        terms_per_label: int = 10,
        candidate_labels: int = 5,
        naive_bayes_alpha: float = 0.05,
        profile_all_labels: bool = False,
    ) -> None:
        if (
            neighbor_k <= 0
            or terms_per_label <= 0
            or candidate_labels <= 0
            or naive_bayes_alpha <= 0
        ):
            raise ValueError("invalid local-evidence configuration")
        self.neighbor_k = neighbor_k
        self.terms_per_label = terms_per_label
        self.candidate_labels = candidate_labels
        self.naive_bayes_alpha = naive_bayes_alpha
        self.profile_all_labels = profile_all_labels
        self.examples: list[LabeledMemoryExample] = []
        self._index: UnicodeTfidfIndex | None = None
        self._statistics_cache: (
            tuple[dict[str, Counter[str]], Counter[str], Counter[str], set[str]] | None
        ) = None

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        self.examples.extend(
            LabeledMemoryExample.from_batch_result(result) for result in batch_results
        )
        self._index = None
        self._statistics_cache = None

    def _ensure_index(self) -> UnicodeTfidfIndex:
        if self._index is None:
            self._index = UnicodeTfidfIndex(example.raw_question for example in self.examples)
        return self._index

    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({example.target for example in self.examples}))

    def extract_query(self, input: str) -> str:
        """Infer and remove a shared task wrapper from training input/raw pairs."""

        prefixes = []
        suffixes = []
        for example in self.examples:
            position = example.input.find(example.raw_question)
            if position < 0:
                continue
            prefixes.append(example.input[:position])
            suffixes.append(example.input[position + len(example.raw_question) :])
        if prefixes and len(set(prefixes)) == 1 and input.startswith(prefixes[0]):
            input = input[len(prefixes[0]) :]
        if suffixes and len(set(suffixes)) == 1 and suffixes[0] and input.endswith(suffixes[0]):
            input = input[: -len(suffixes[0])]
        return input

    def _statistics(
        self,
    ) -> tuple[dict[str, Counter[str]], Counter[str], Counter[str], set[str]]:
        if self._statistics_cache is not None:
            return self._statistics_cache
        by_label: dict[str, Counter[str]] = defaultdict(Counter)
        label_frequency: Counter[str] = Counter()
        global_counts: Counter[str] = Counter()
        vocabulary: set[str] = set()
        for example in self.examples:
            counts = classification_terms(example.raw_question)
            by_label[example.target].update(counts)
            label_frequency[example.target] += 1
            global_counts.update(counts)
            vocabulary.update(counts)
        self._statistics_cache = by_label, label_frequency, global_counts, vocabulary
        return self._statistics_cache

    def ranked_label_candidates(self, query: str) -> tuple[tuple[str, float], ...]:
        if not self.examples:
            return ()
        by_label, label_frequency, _global_counts, vocabulary = self._statistics()
        totals = {label: sum(counts.values()) for label, counts in by_label.items()}
        # The upstream MemorySystem receives a full task template at inference time while
        # training examples expose a shorter raw_question. Ignore out-of-vocabulary wrapper
        # terms so repeated benchmark instructions cannot bias label likelihoods through the
        # class-specific denominator.
        query_counts = Counter(
            {
                term: frequency
                for term, frequency in classification_terms(query).items()
                if term in vocabulary
            }
        )
        scores = {}
        for label in self.labels():
            denominator = totals[label] + self.naive_bayes_alpha * len(vocabulary)
            scores[label] = math.log(label_frequency[label] / len(self.examples)) + sum(
                frequency
                * math.log(
                    (by_label[label][term] + self.naive_bayes_alpha) / denominator
                )
                for term, frequency in query_counts.items()
            )
        ranked = sorted(scores, key=lambda label: (-scores[label], label))
        best = scores[ranked[0]]
        return tuple((label, scores[label] - best) for label in ranked[: self.candidate_labels])

    def _profile(self, label: str) -> tuple[str, ...]:
        by_label, _frequency, global_counts, _vocabulary = self._statistics()
        own = by_label[label]

        def score(term: str) -> float:
            other = global_counts[term] - own[term]
            return math.log((own[term] + 0.1) / (other + 0.1)) * math.sqrt(own[term])

        useful = [term for term in own if len(term) > 1 and not term.isdigit()]
        return tuple(sorted(useful, key=lambda term: (-score(term), term))[: self.terms_per_label])

    def build_context(self, input: str) -> str:
        if not self.examples:
            return "(no labeled memory available)"
        candidates = self.ranked_label_candidates(input)
        candidate_block = "\n".join(
            f"- {label}: relative log-score {score:.3f}; "
            f"cues: {', '.join(self._profile(label))}"
            for label, score in candidates
        )
        neighbors = self._ensure_index().rank(input)[: self.neighbor_k]
        neighbor_block = "\n\n".join(
            f"Q: {self.examples[index].raw_question}\nA: {self.examples[index].target}"
            for index in neighbors
        )
        profile_block = ""
        if self.profile_all_labels:
            profile_block = "\n\nAll learned label cue profiles:\n" + "\n".join(
                f"- {label}: {', '.join(self._profile(label))}" for label in self.labels()
            )
        return (
            "Data-derived candidate ranking (a fallible signal, not the answer):\n"
            + candidate_block
            + "\n\nNearest labeled cases:\n"
            + neighbor_block
            + profile_block
        )

    def build_prompt(self, input: str) -> str:
        context = self.build_context(input)
        return f"""Solve this classification problem using compact evidence learned only from the
training split. Compare the complete fact pattern, use the candidate scores only as one signal,
and distinguish labels using both positive and missing cues.

{context}

Problem:
{input}

Return valid JSON with exactly these fields:
{{"reasoning": "brief evidence-based reasoning", "final_answer": "exact final answer"}}"""

    def get_state(self) -> str:
        return json.dumps(
            {
                "schema_version": "1",
                "kind": "spiral-local-evidence",
                "neighbor_k": self.neighbor_k,
                "terms_per_label": self.terms_per_label,
                "candidate_labels": self.candidate_labels,
                "naive_bayes_alpha": self.naive_bayes_alpha,
                "profile_all_labels": self.profile_all_labels,
                "examples": [asdict(example) for example in self.examples],
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def set_state(self, state: str) -> None:
        payload = json.loads(state)
        if payload.get("kind") != "spiral-local-evidence":
            raise ValueError("not a Spiral local-evidence state")
        self.neighbor_k = int(payload["neighbor_k"])
        self.terms_per_label = int(payload["terms_per_label"])
        self.candidate_labels = int(payload["candidate_labels"])
        self.naive_bayes_alpha = float(payload["naive_bayes_alpha"])
        self.profile_all_labels = bool(payload.get("profile_all_labels", False))
        self.examples = [LabeledMemoryExample(**example) for example in payload["examples"]]
        self._index = None
        self._statistics_cache = None


class SpiralCandidateAdjudicationMemory:
    """Compose independent candidates and compact evidence in a final model call."""

    def __init__(
        self,
        *,
        paper_context_char_budget: int = 100_000,
        paper_contrastive_pairs: int = 8,
        local_neighbor_k: int = 12,
    ) -> None:
        self.paper = LabelPrimedQueryMemory(
            context_char_budget=paper_context_char_budget,
            contrastive_pairs=paper_contrastive_pairs,
        )
        self.local = SpiralLocalEvidenceMemory(neighbor_k=local_neighbor_k)

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        self.paper.learn_from_batch(batch_results)
        self.local.learn_from_batch(batch_results)

    def build_adjudication_prompt(
        self,
        input: str,
        model_only_candidate: str,
        label_primed_candidate: str,
    ) -> str:
        evidence = self.local.build_context(input)
        return f"""Adjudicate two independently produced answers to the same classification problem.

Candidate A (model-only, strongest prior when memory lacks the label):
{model_only_candidate}

Candidate B (full label-primed coverage and contrastive retrieval):
{label_primed_candidate}

Compact training-derived evidence for adjudication:
{evidence}

Problem:
{input}

Decision rules:
- Decide from the complete facts; candidates and statistical scores can each be wrong.
- When only spelling or taxonomy differs, use the exact observed label spelling that matches
  the intended class.
- Do not union candidate labels. Include each label only when the facts independently support it.
- A label absent from training memory remains allowed when the facts clearly require it.
- Put only the exact task answer in final_answer, following the problem's requested format.

Return valid JSON with exactly these fields:
{{"reasoning": "brief adjudication", "final_answer": "exact final answer"}}"""

    def get_state(self) -> str:
        return json.dumps(
            {
                "schema_version": "1",
                "kind": "spiral-candidate-adjudication",
                "paper": json.loads(self.paper.get_state()),
                "local": json.loads(self.local.get_state()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def set_state(self, state: str) -> None:
        payload = json.loads(state)
        if payload.get("kind") != "spiral-candidate-adjudication":
            raise ValueError("not a Spiral candidate-adjudication state")
        self.paper.set_state(json.dumps(payload["paper"], ensure_ascii=False))
        self.local.set_state(json.dumps(payload["local"], ensure_ascii=False))


__all__ = ["SpiralCandidateAdjudicationMemory", "SpiralLocalEvidenceMemory"]
