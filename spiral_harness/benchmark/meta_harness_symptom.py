"""Pinned Meta-Harness Symptom2Disease evaluation with retrieval harnesses."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from spiral_harness.benchmark.gsm8k_smoke import build_live_gsm8k_spec
from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.core.models import HARNESS_MANIFEST_MEDIA_TYPE
from spiral_harness.execution.contracts import ModelRequest, ResolvedHarness
from spiral_harness.execution.model import ModelBackend
from spiral_harness.storage.artifact_store import ArtifactStore

META_HARNESS_REVISION = "44b9942127847f7421db70d8c7e48407f09a3c70"
SPLIT_SHA256 = {
    "train": "b41e494b40bf971a1eadd79ed4dda527640916abe0a2dc94e047e13ef293ee4c",
    "val": "f6ad74bea9fb59f4a3abf930668042c33fbb7519bc9e2568a9758c2a0b1277e8",
    "test": "6a6b224665715f40d42a46eed82498698d39fa7689241eaae2cc2276e444671d",
}
REFERENCE_ACCURACY = 0.83
LABELS = (
    "drug reaction",
    "allergy",
    "chicken pox",
    "diabetes",
    "psoriasis",
    "hypertension",
    "cervical spondylosis",
    "bronchial asthma",
    "varicose veins",
    "malaria",
    "dengue",
    "arthritis",
    "impetigo",
    "fungal infection",
    "common cold",
    "gastroesophageal reflux disease",
    "urinary tract infection",
    "typhoid",
    "pneumonia",
    "peptic ulcer disease",
    "jaundice",
    "migraine",
)
_WORDS = re.compile(r"[a-z]+")
_STOP = frozenset(
    {
        "a",
        "all",
        "also",
        "am",
        "an",
        "and",
        "are",
        "been",
        "feel",
        "feeling",
        "for",
        "from",
        "get",
        "gets",
        "got",
        "had",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "lately",
        "like",
        "lot",
        "me",
        "my",
        "of",
        "on",
        "or",
        "really",
        "that",
        "the",
        "this",
        "time",
        "to",
        "very",
        "was",
        "were",
        "when",
        "with",
        "worse",
    }
)


@dataclass(frozen=True, slots=True)
class SymptomExample:
    question: str
    answer: str


def load_split(path: Path, split: str) -> tuple[SymptomExample, ...]:
    data = Path(path).read_bytes()
    if sha256_bytes(data) != SPLIT_SHA256[split]:
        raise ValueError(f"Meta-Harness {split} split hash mismatch")
    return tuple(SymptomExample(**json.loads(line)) for line in data.decode().splitlines())


class SymptomRetriever:
    def __init__(self, examples: tuple[SymptomExample, ...]) -> None:
        self.examples = examples
        docs = [self._tokens(item.question) for item in examples]
        counts = Counter(word for doc in docs for word in doc)
        self.idf = {
            word: math.log((len(docs) + 1) / (count + 1)) + 1 for word, count in counts.items()
        }
        self.vectors = tuple(self._vector(doc) for doc in docs)

    def label_profiles(self, terms_per_label: int = 10) -> str:
        """Build deterministic discriminative term profiles from training data only."""

        by_label: dict[str, Counter[str]] = {}
        for example in self.examples:
            counts = by_label.setdefault(example.answer, Counter())
            counts.update(self._tokens(example.question))
        lines = []
        for label in LABELS:
            counts = by_label.get(label, Counter())
            ranked = sorted(
                counts,
                key=lambda word: (-counts[word] * self.idf.get(word, 1.0), word),
            )[:terms_per_label]
            lines.append(f"- {label}: {', '.join(ranked)}")
        return "\n".join(lines)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(_WORDS.findall(text.lower())) - _STOP

    def _vector(self, tokens: set[str]) -> dict[str, float]:
        return {word: self.idf.get(word, 1.0) for word in tokens}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        dot = sum(value * right.get(word, 0.0) for word, value in left.items())
        norms = math.sqrt(sum(x * x for x in left.values()) * sum(x * x for x in right.values()))
        return dot / norms if norms else 0.0

    def retrieve(self, question: str, limit: int) -> tuple[SymptomExample, ...]:
        query = self._vector(self._tokens(question))
        ranked = sorted(
            range(len(self.examples)),
            key=lambda index: (-self._cosine(query, self.vectors[index]), index),
        )
        return tuple(self.examples[index] for index in ranked[:limit])


def parse_diagnosis(text: str) -> str:
    match = re.search(r"\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]", text, re.I | re.S)
    value = match.group(1) if match else text
    normalized = re.sub(r"[.!?]+$", "", " ".join(value.lower().strip().split()))
    # Some OpenAI-compatible providers wrap the requested tagged answer in one
    # additional pair of brackets. Remove it only when the enclosed value is an
    # exact member of the benchmark label ontology; arbitrary output is untouched.
    if normalized.startswith("[") and normalized.endswith("]"):
        enclosed = normalized[1:-1].strip()
        if enclosed in LABELS:
            return enclosed
    return normalized


def build_prompt(
    question: str, examples: tuple[SymptomExample, ...], label_profiles: str = ""
) -> str:
    demonstrations = "\n\n".join(
        f"Symptoms: {item.question}\nDiagnosis: {item.answer}" for item in examples
    )
    labels = ", ".join(LABELS)
    return (
        "Classify the patient into exactly one allowed diagnosis. Similar labeled cases are "
        "retrieved from the official training split; use them as evidence, but distinguish close "
        "conditions by the complete symptom pattern.\n\nAllowed diagnoses: "
        + labels
        + "\n\nTraining-derived label profiles (discriminative terms, not medical definitions):\n"
        + label_profiles
        + "\n\nRetrieved cases:\n"
        + demonstrations
        + "\n\nPatient: "
        + question
        + "\n\nReturn only [DIAGNOSIS]exact allowed label[/DIAGNOSIS]."
    )


def build_pure_prompt(question: str) -> str:
    """Build the no-memory arm without training examples or derived profiles."""

    return (
        "Based on the patient's symptoms, select exactly one diagnosis from this allowed list: "
        + ", ".join(LABELS)
        + "\n\nPatient: "
        + question
        + "\n\nReturn only [DIAGNOSIS]exact allowed label[/DIAGNOSIS]."
    )


def run_symptom_evaluation(
    *,
    train_path: Path,
    evaluation_path: Path,
    split: str,
    output: Path,
    backend: ModelBackend,
    model: str,
    arm: str = "evolved",
    retrieval_k: int = 8,
    max_output_tokens: int = 128,
    timeout_seconds: float = 90.0,
) -> dict[str, object]:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if arm not in {"pure", "evolved"}:
        raise ValueError("arm must be pure or evolved")
    train = load_split(train_path, "train")
    evaluation = load_split(evaluation_path, split)
    retriever = SymptomRetriever(train)
    label_profiles = retriever.label_profiles()
    store = ArtifactStore(Path(output) / "artifacts")
    system = "You are a careful medical text classifier. Follow the output contract exactly."
    harness_ref = store.put_json(
        {
            "schema_version": "1",
            "kind": "retrieval-classifier",
            "arm": arm,
            "retrieval_k": retrieval_k,
            "prompt_builder_sha256": sha256_bytes(build_prompt.__code__.co_code),
        },
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    harness = ResolvedHarness.from_prompt(harness_ref=harness_ref, system_prompt=system)
    spec = build_live_gsm8k_spec(
        backend_fingerprint=backend.fingerprint,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    records = []
    for index, item in enumerate(evaluation):
        prompt = (
            build_pure_prompt(item.question)
            if arm == "pure"
            else build_prompt(
                item.question,
                retriever.retrieve(item.question, retrieval_k),
                label_profiles,
            )
        )
        response = backend.invoke(
            spec=spec,
            request=ModelRequest(
                task_id=f"meta-harness/Symptom2Disease/{split}/{index}",
                harness_ref=harness.harness_ref,
                base_system_prompt=harness.base_system_prompt,
                base_system_prompt_sha256=harness.base_system_prompt_sha256,
                system_prompt=harness.system_prompt,
                resolved_prompt_sha256=harness.resolved_prompt_sha256,
                user_prompt=prompt,
                seed=index,
            ),
        )
        prediction = parse_diagnosis(response.output)
        records.append(
            {
                "index": index,
                "prediction": prediction,
                "target": item.answer,
                "correct": prediction == item.answer,
                "response_sha256": sha256_bytes(response.output.encode()),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )
    correct = sum(bool(item["correct"]) for item in records)
    payload: dict[str, object] = {
        "schema_version": "1",
        "benchmark": "Meta-Harness/Symptom2Disease",
        "revision": META_HARNESS_REVISION,
        "split": split,
        "model": model,
        "arm": arm,
        "retrieval_k": retrieval_k,
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "reference_accuracy": REFERENCE_ACCURACY,
        "exceeds_reference": correct / len(records) > REFERENCE_ACCURACY,
        "total_tokens": sum(int(x["input_tokens"]) + int(x["output_tokens"]) for x in records),
        "records": records,
    }
    ref = store.put_json(
        payload, media_type="application/vnd.spiral-harness.meta-harness-symptom.v1+json"
    )
    payload["artifact_sha256"] = ref.sha256
    return payload


__all__ = [
    "META_HARNESS_REVISION",
    "REFERENCE_ACCURACY",
    "SymptomRetriever",
    "build_prompt",
    "build_pure_prompt",
    "load_split",
    "parse_diagnosis",
    "run_symptom_evaluation",
]
