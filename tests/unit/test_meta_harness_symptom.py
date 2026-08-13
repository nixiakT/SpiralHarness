from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import spiral_harness.benchmark.meta_harness_symptom as subject


def _split(tmp_path: Path, examples: list[dict[str, str]]) -> Path:
    path = tmp_path / "split.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in examples))
    return path


def test_load_split_requires_exact_pinned_hash(tmp_path: Path, monkeypatch) -> None:
    path = _split(tmp_path, [{"question": "itchy skin", "answer": "allergy"}])
    monkeypatch.setitem(subject.SPLIT_SHA256, "val", hashlib.sha256(path.read_bytes()).hexdigest())

    assert subject.load_split(path, "val")[0].answer == "allergy"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        subject.load_split(path, "val")


def test_retriever_prefers_shared_discriminative_symptoms() -> None:
    retriever = subject.SymptomRetriever(
        (
            subject.SymptomExample("itchy red skin rash", "allergy"),
            subject.SymptomExample("burning stomach heartburn", "peptic ulcer disease"),
        )
    )

    assert retriever.retrieve("red itchy rash", 1)[0].answer == "allergy"
    profiles = retriever.label_profiles(terms_per_label=2)
    assert "- allergy:" in profiles
    assert "itchy" in profiles


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[DIAGNOSIS]Diabetes[/DIAGNOSIS]", "diabetes"),
        ("[[DIAGNOSIS] dengue [/DIAGNOSIS]]", "dengue"),
        ("  common cold. ", "common cold"),
    ],
)
def test_parse_diagnosis(text: str, expected: str) -> None:
    assert subject.parse_diagnosis(text) == expected


def test_prompt_discloses_only_training_examples() -> None:
    prompt = subject.build_prompt(
        "new patient",
        (subject.SymptomExample("known symptoms", "malaria"),),
        "- malaria: fever, chills",
    )

    assert "known symptoms" in prompt
    assert "malaria" in prompt
    assert "new patient" in prompt
    assert "exact allowed label" in prompt
    assert "fever, chills" in prompt


def test_pure_prompt_contains_no_training_evidence() -> None:
    prompt = subject.build_pure_prompt("new patient")

    assert "new patient" in prompt
    assert "Retrieved cases" not in prompt
    assert "Training-derived" not in prompt
