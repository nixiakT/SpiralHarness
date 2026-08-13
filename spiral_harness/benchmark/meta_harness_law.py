"""Same-model pure and retrieval arms for pinned Meta-Harness LawBench."""

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

SPLIT_SHA256 = {
    "train": "36ead48b5e88e02f6bf5944d1c00c561f6592e5573f9ecf05c23affc111c386c",
    "val": "c5d900d64a7be5359579e4c2fc0b499f18ebbe9a96fee1ff43879d511bc992ea",
    "test": "a08a9aba4da6814fc0efdfc713814a2562f4ce1d174981d8d5823a126b99f6dc",
}


@dataclass(frozen=True, slots=True)
class LawExample:
    question: str
    answer: str


def _clean_answer(answer: str) -> str:
    return answer.split("罪名:")[-1].strip() if "罪名:" in answer else answer.strip()


def load_split(path: Path, split: str) -> tuple[LawExample, ...]:
    data = Path(path).read_bytes()
    if sha256_bytes(data) != SPLIT_SHA256[split]:
        raise ValueError(f"Meta-Harness LawBench {split} split hash mismatch")
    return tuple(
        LawExample(question=item["question"], answer=_clean_answer(item["answer"]))
        for item in map(json.loads, data.decode().splitlines())
    )


def parse_charges(text: str) -> frozenset[str]:
    match = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", text, re.S)
    value = match.group(1).strip() if match else text.strip()
    value = re.sub(r"<eoa>.*", "", value).strip()
    return frozenset(
        x.strip()
        for x in re.split("[;\N{FULLWIDTH SEMICOLON},\N{FULLWIDTH COMMA}、]", value)
        if x.strip()
    )


class LawRetriever:
    def __init__(self, examples: tuple[LawExample, ...]) -> None:
        self.examples = examples
        docs = [self._grams(item.question) for item in examples]
        df = Counter(gram for doc in docs for gram in doc)
        self.idf = {g: math.log((len(docs) + 1) / (n + 1)) + 1 for g, n in df.items()}
        self.vectors = tuple({g: self.idf[g] for g in doc} for doc in docs)

    @staticmethod
    def _grams(text: str) -> set[str]:
        normalized = re.sub(r"\s+", "", text)
        return {normalized[i : i + 2] for i in range(max(0, len(normalized) - 1))}

    def retrieve(self, question: str, limit: int) -> tuple[LawExample, ...]:
        query = {g: self.idf.get(g, 1.0) for g in self._grams(question)}

        def score(index: int) -> float:
            vector = self.vectors[index]
            dot = sum(v * vector.get(g, 0.0) for g, v in query.items())
            norm = math.sqrt(
                sum(v * v for v in query.values()) * sum(v * v for v in vector.values())
            )
            return dot / norm if norm else 0.0

        ranked = sorted(range(len(self.examples)), key=lambda i: (-score(i), i))
        return tuple(self.examples[i] for i in ranked[:limit])


def build_prompt(question: str, arm: str, examples: tuple[LawExample, ...] = ()) -> str:
    evidence = ""
    if arm == "evolved":
        evidence = "\n\n训练集中的相似判例:\n" + "\n\n".join(
            f"案件:{item.question}\n罪名:{item.answer}" for item in examples
        )
    return (
        "请依据案件事实判断罪名。可能存在多个罪名;不要输出训练判例中没有证据支持的罪名。"
        + evidence
        + "\n\n待判断案件:\n"
        + question
        + "\n\n只输出:[罪名]罪名1;罪名2<eoa>。单个罪名不要加分号。"
    )


def run_law_evaluation(
    *,
    train_path: Path,
    evaluation_path: Path,
    split: str,
    output: Path,
    backend: ModelBackend,
    model: str,
    arm: str,
    retrieval_k: int = 8,
    max_output_tokens: int = 128,
    timeout_seconds: float = 90.0,
) -> dict[str, object]:
    if split not in {"val", "test"} or arm not in {"pure", "evolved"}:
        raise ValueError("invalid split or arm")
    train, evaluation = load_split(train_path, "train"), load_split(evaluation_path, split)
    retriever = LawRetriever(train)
    store = ArtifactStore(Path(output) / "artifacts")
    system = "你是严格遵循输出格式的中国刑法文本分类器。"
    href = store.put_json(
        {"schema_version": "1", "benchmark": "LawBench", "arm": arm, "retrieval_k": retrieval_k},
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    harness = ResolvedHarness.from_prompt(harness_ref=href, system_prompt=system)
    spec = build_live_gsm8k_spec(
        backend_fingerprint=backend.fingerprint,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    records = []
    for index, item in enumerate(evaluation):
        examples = retriever.retrieve(item.question, retrieval_k) if arm == "evolved" else ()
        response = backend.invoke(
            spec=spec,
            request=ModelRequest(
                task_id=f"meta-harness/LawBench/{split}/{index}",
                harness_ref=href,
                base_system_prompt=harness.base_system_prompt,
                base_system_prompt_sha256=harness.base_system_prompt_sha256,
                system_prompt=harness.system_prompt,
                resolved_prompt_sha256=harness.resolved_prompt_sha256,
                user_prompt=build_prompt(item.question, arm, examples),
                seed=index,
            ),
        )
        prediction, target = parse_charges(response.output), parse_charges(item.answer)
        records.append(
            {
                "index": index,
                "prediction": sorted(prediction),
                "target": sorted(target),
                "correct": prediction == target,
                "response_sha256": sha256_bytes(response.output.encode()),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )
    correct = sum(bool(x["correct"]) for x in records)
    payload: dict[str, object] = {
        "schema_version": "1",
        "benchmark": "Meta-Harness/LawBench",
        "split": split,
        "model": model,
        "arm": arm,
        "retrieval_k": retrieval_k,
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "total_tokens": sum(int(x["input_tokens"]) + int(x["output_tokens"]) for x in records),
        "records": records,
    }
    ref = store.put_json(
        payload, media_type="application/vnd.spiral-harness.meta-harness-law.v1+json"
    )
    payload["artifact_sha256"] = ref.sha256
    return payload


__all__ = [
    "LawExample",
    "LawRetriever",
    "build_prompt",
    "load_split",
    "parse_charges",
    "run_law_evaluation",
]
