from __future__ import annotations

import json

import pytest

from spiral_harness.benchmark.meta_harness_paper import (
    LabelPrimedQueryMemory,
    SpiralDraftVerificationMemory,
    UnicodeTfidfIndex,
    atomic_labels,
    clean_classification_answer,
    route_by_majority,
    route_by_taxonomy_specificity,
    route_by_validation_score_band,
    unicode_terms,
)
from spiral_harness.benchmark.meta_harness_spiral import (
    SpiralCandidateAdjudicationMemory,
    SpiralLocalEvidenceMemory,
)


def _result(question: str, target: str) -> dict[str, object]:
    return {
        "input": f"task wrapper: {question}",
        "raw_question": question,
        "ground_truth": target,
    }


def test_unicode_terms_support_english_words_and_han_bigrams() -> None:
    terms = unicode_terms("Itchy SKIN; 秘密窃取财物")

    assert terms["itchy"] == 1
    assert terms["skin"] == 1
    assert terms["窃取"] == 1


def test_tfidf_prefers_relevant_document_in_both_languages() -> None:
    index = UnicodeTfidfIndex(("itchy red rash", "burning stomach", "秘密窃取财物"))

    assert index.rank("itchy rash")[0] == 0
    assert index.rank("窃取他人财物")[0] == 2


def test_atomic_labels_does_not_split_chinese_list_punctuation() -> None:
    assert atomic_labels("盗窃;伪造、变造居民身份证") == (
        "盗窃",
        "伪造、变造居民身份证",
    )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("[DIAGNOSIS]dengue[/DIAGNOSIS]", "dengue"),
        ("[罪名]盗窃;诈骗<eoa>", "盗窃;诈骗"),
        ("[罪名]放火<eoa]", "放火"),
        ("unwrapped label", "unwrapped label"),
    ],
)
def test_clean_classification_answer_only_removes_output_wrappers(
    answer: str, expected: str
) -> None:
    assert clean_classification_answer(answer) == expected


def test_score_band_router_uses_frozen_margin_and_agreement_rules() -> None:
    assert (
        route_by_validation_score_band(
            local_answer="label-local",
            label_primed_answer="label-paper",
            ranked_candidates=(("label-nb", 0.0), ("other", -0.5)),
        )
        == "label-nb"
    )
    assert (
        route_by_validation_score_band(
            local_answer="label-local",
            label_primed_answer="label-nb",
            ranked_candidates=(("label-nb", 0.0), ("other", -20.0)),
        )
        == "label-nb"
    )
    assert (
        route_by_validation_score_band(
            local_answer="label-local",
            label_primed_answer="label-paper",
            ranked_candidates=(("label-nb", 0.0), ("other", -12.0)),
        )
        == "label-local"
    )


def test_taxonomy_router_prefers_specific_or_non_union_label() -> None:
    assert (
        route_by_taxonomy_specificity(
            model_only_answer="侵犯公民个人信息;诈骗",
            draft_verified_answer="侵犯公民个人信息;诈骗",
            label_primed_answer="非法获取公民个人信息;诈骗",
        )
        == "非法获取公民个人信息;诈骗"
    )
    assert (
        route_by_taxonomy_specificity(
            model_only_answer="交通肇事",
            draft_verified_answer="交通肇事;过失致人死亡",
            label_primed_answer="过失致人死亡",
        )
        == "过失致人死亡"
    )
    assert (
        route_by_taxonomy_specificity(
            model_only_answer="销售假冒注册商标的商品",
            draft_verified_answer="销售假冒注册商标的商品",
            label_primed_answer="假冒注册商标",
        )
        == "销售假冒注册商标的商品"
    )
    assert (
        route_by_taxonomy_specificity(
            model_only_answer="盗窃",
            draft_verified_answer="盗窃;破坏广播电视设施",
            label_primed_answer="盗窃;破坏电力设备",
            local_evidence_answer="破坏电力设备",
        )
        == "破坏电力设备"
    )


def test_majority_router_normalizes_wrappers_and_uses_fallback_for_tie() -> None:
    assert (
        route_by_majority(
            ("[DIAGNOSIS]dengue[/DIAGNOSIS]", "dengue", "malaria"),
            fallback="malaria",
        )
        == "dengue"
    )
    assert route_by_majority(("dengue", "malaria"), fallback="malaria") == "malaria"


def test_label_primed_context_has_coverage_and_contrastive_evidence() -> None:
    memory = LabelPrimedQueryMemory(contrastive_pairs=2)
    memory.learn_from_batch(
        [
            _result("itchy red skin rash", "allergy"),
            _result("itchy circular skin rash", "fungal infection"),
            _result("burning stomach after meals", "peptic ulcer disease"),
        ]
    )

    context = memory.build_context("red itchy skin")

    assert "allergy | fungal infection | peptic ulcer disease" in context
    assert context.count("[Coverage label:") == 3
    assert "Local contrastive pair" in context
    assert "VERSUS" in context


def test_label_primed_context_can_include_multiple_representatives_per_label() -> None:
    memory = LabelPrimedQueryMemory(coverage_per_label=2, contrastive_pairs=0)
    memory.learn_from_batch(
        [
            _result("itchy red skin", "allergy"),
            _result("watery eyes and sneezing", "allergy"),
            _result("burning stomach", "peptic ulcer disease"),
        ]
    )

    context = memory.build_context("itchy watery eyes")

    assert context.count("[Coverage label: allergy]") == 2
    assert context.count("[Coverage label: peptic ulcer disease]") == 1


def test_label_primed_prompt_preserves_official_problem_and_json_contract() -> None:
    memory = LabelPrimedQueryMemory()
    memory.learn_from_batch([_result("known case", "label-a")])

    prompt = memory.build_prompt("OFFICIAL DATASET PROMPT")

    assert "OFFICIAL DATASET PROMPT" in prompt
    assert '"final_answer"' in prompt
    assert "paper" not in prompt.lower()


def test_state_round_trip_discloses_reconstruction_boundary() -> None:
    original = LabelPrimedQueryMemory(context_char_budget=1234, contrastive_pairs=3)
    original.learn_from_batch([_result("known case", "label-a")])
    state = original.get_state()
    restored = LabelPrimedQueryMemory()
    restored.set_state(state)

    payload = json.loads(state)
    assert payload["kind"] == "meta-harness-paper-algorithm-reconstruction"
    assert restored.get_state() == state


def test_invalid_state_fails_closed() -> None:
    memory = LabelPrimedQueryMemory()

    with pytest.raises(ValueError, match="not a Meta-Harness"):
        memory.set_state('{"kind": "official-source-reproduction"}')


def test_spiral_draft_verification_keeps_prior_and_local_counterevidence() -> None:
    memory = SpiralDraftVerificationMemory(neighbor_k=1, confirmer_k=1)
    memory.learn_from_batch(
        [
            _result("itchy red skin rash", "allergy"),
            _result("itchy circular rash", "fungal infection"),
            _result("burning stomach", "peptic ulcer disease"),
        ]
    )

    prompt = memory.build_verification_prompt(
        "itchy circular skin problem", "[DIAGNOSIS]allergy[/DIAGNOSIS]"
    )

    assert "First-pass draft" in prompt
    assert "allergy" in prompt
    assert "fungal infection" in prompt
    assert "labels absent from" in prompt
    assert "exact task answer" in prompt


def test_spiral_draft_state_round_trip() -> None:
    original = SpiralDraftVerificationMemory(neighbor_k=9, confirmer_k=2)
    original.learn_from_batch([_result("known", "label")])
    restored = SpiralDraftVerificationMemory()
    restored.set_state(original.get_state())

    assert restored.get_state() == original.get_state()


def test_spiral_local_evidence_exposes_ranked_signal_and_nearest_cases() -> None:
    memory = SpiralLocalEvidenceMemory(neighbor_k=2, candidate_labels=2)
    memory.learn_from_batch(
        [
            _result("itchy red rash on skin", "allergy"),
            _result("itchy circular rash on skin", "fungal infection"),
            _result("burning stomach pain", "peptic ulcer disease"),
        ]
    )

    prompt = memory.build_prompt("itchy circular skin rash")
    candidates = memory.ranked_label_candidates("itchy circular skin rash")

    assert candidates[0][0] == "fungal infection"
    assert "fallible signal" in prompt
    assert "Nearest labeled cases" in prompt
    assert "exact final answer" in prompt


def test_local_evidence_ignores_unseen_task_wrapper_terms() -> None:
    memory = SpiralLocalEvidenceMemory(neighbor_k=1, candidate_labels=2)
    memory.learn_from_batch(
        [
            _result("itchy circular rash", "fungal infection"),
            _result("burning stomach pain", "peptic ulcer disease"),
        ]
    )

    raw = memory.ranked_label_candidates("itchy circular rash")
    wrapped = memory.ranked_label_candidates(
        "UNSEEN TEMPLATE BOILERPLATE exact JSON reasoning final answer itchy circular rash"
    )

    assert wrapped == raw


def test_local_evidence_infers_shared_task_wrapper_without_dataset_markers() -> None:
    memory = SpiralLocalEvidenceMemory()
    memory.learn_from_batch(
        [
            {
                "input": "SHARED PREFIX\nfirst query\nSHARED SUFFIX",
                "raw_question": "first query",
                "ground_truth": "a",
            },
            {
                "input": "SHARED PREFIX\nsecond query\nSHARED SUFFIX",
                "raw_question": "second query",
                "ground_truth": "b",
            },
        ]
    )

    assert memory.extract_query("SHARED PREFIX\nnew query\nSHARED SUFFIX") == "new query"


def test_spiral_local_evidence_state_round_trip() -> None:
    original = SpiralLocalEvidenceMemory(neighbor_k=3, naive_bayes_alpha=0.2)
    original.learn_from_batch([_result("known", "label")])
    restored = SpiralLocalEvidenceMemory()
    restored.set_state(original.get_state())

    assert restored.get_state() == original.get_state()


def test_local_evidence_can_expose_all_training_derived_label_profiles() -> None:
    memory = SpiralLocalEvidenceMemory(
        neighbor_k=1, candidate_labels=1, profile_all_labels=True
    )
    memory.learn_from_batch(
        [
            _result("itchy circular rash", "fungal infection"),
            _result("burning stomach pain", "peptic ulcer disease"),
        ]
    )

    context = memory.build_context("itchy circular rash")

    assert "All learned label cue profiles" in context
    assert "fungal infection" in context
    assert "peptic ulcer disease" in context


def test_candidate_adjudication_preserves_independent_candidates_and_evidence() -> None:
    memory = SpiralCandidateAdjudicationMemory(local_neighbor_k=1)
    memory.learn_from_batch(
        [
            _result("itchy red rash", "allergy"),
            _result("itchy circular rash", "fungal infection"),
        ]
    )

    prompt = memory.build_adjudication_prompt(
        "itchy circular rash", "allergy", "fungal infection"
    )

    assert "Candidate A" in prompt
    assert "Candidate B" in prompt
    assert "allergy" in prompt
    assert "fungal infection" in prompt
    assert "Do not union" in prompt


def test_candidate_adjudication_state_round_trip() -> None:
    original = SpiralCandidateAdjudicationMemory(local_neighbor_k=3)
    original.learn_from_batch([_result("known", "label")])
    restored = SpiralCandidateAdjudicationMemory()
    restored.set_state(original.get_state())

    assert restored.get_state() == original.get_state()
