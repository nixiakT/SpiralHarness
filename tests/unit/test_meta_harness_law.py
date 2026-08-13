from __future__ import annotations

import spiral_harness.benchmark.meta_harness_law as subject


def test_parse_charges_is_order_insensitive() -> None:
    assert subject.parse_charges("[罪名]盗窃;诈骗<eoa>") == frozenset({"盗窃", "诈骗"})


def test_law_retriever_prefers_shared_chinese_bigrams() -> None:
    retriever = subject.LawRetriever(
        (
            subject.LawExample("秘密窃取他人财物", "盗窃"),
            subject.LawExample("故意伤害他人身体", "故意伤害"),
        )
    )
    assert retriever.retrieve("窃取财物", 1)[0].answer == "盗窃"


def test_pure_prompt_has_no_training_cases() -> None:
    prompt = subject.build_prompt("案件事实", "pure")
    assert "相似判例" not in prompt


def test_evolved_prompt_discloses_retrieved_training_case() -> None:
    prompt = subject.build_prompt("案件事实", "evolved", (subject.LawExample("训练案件", "盗窃"),))
    assert "训练案件" in prompt
    assert "盗窃" in prompt


def test_baseline_prompt_discloses_fixed_training_case() -> None:
    prompt = subject.build_prompt("案件事实", "baseline", (subject.LawExample("固定案件", "诈骗"),))
    assert "固定案件" in prompt
