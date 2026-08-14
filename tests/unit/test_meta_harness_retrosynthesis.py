from __future__ import annotations

import json

import pytest

from spiral_harness.benchmark.meta_harness_retrosynthesis import (
    SpiralRetrosynthesisMemory,
    compile_retrosynthesis_answer,
    parse_retrosynthesis_query,
    smiles_terms,
    weighted_jaccard,
)


def _query(reaction_type: str, product: str) -> str:
    return (
        f"Context: The reaction type is {reaction_type}.\n"
        f"Input: {product}\nAnswer: "
    )


def _result(reaction_type: str, product: str, target: str) -> dict[str, object]:
    question = _query(reaction_type, product)
    return {
        "input": f"official wrapper\n{question}",
        "raw_question": question,
        "ground_truth": target,
    }


def test_parse_retrosynthesis_query_extracts_family_and_product() -> None:
    assert parse_retrosynthesis_query(_query("Oxidations", "CC=O")) == (
        "Oxidations",
        "CC=O",
    )

    with pytest.raises(ValueError, match="not a Meta-Harness"):
        parse_retrosynthesis_query("ordinary classification query")


def test_smiles_similarity_prefers_shared_local_structure() -> None:
    query = smiles_terms("COc1ccc(CBr)cc1")
    close = smiles_terms("N#Cc1ccc(CBr)cc1Br")
    far = smiles_terms("CC1NCCC1")

    assert weighted_jaccard(query, close) > weighted_jaccard(query, far)


def test_retrieval_filters_reaction_family_before_structure_ranking() -> None:
    memory = SpiralRetrosynthesisMemory(neighbor_k=2)
    memory.learn_from_batch(
        [
            _result("Functional group addition (FGA)", "N#Cc1ccc(CBr)cc1Br", "FGA-A"),
            _result("Functional group addition (FGA)", "CC(C)Br", "FGA-B"),
            _result("Oxidations", "COc1ccc(CBr)cc1", "OXIDATION"),
        ]
    )

    selected = memory.retrieve(_query("Functional group addition (FGA)", "COc1ccc(CBr)cc1"))

    assert len(selected) == 2
    assert {example.target for example, _score in selected} == {"FGA-A", "FGA-B"}
    assert selected[0][0].target == "FGA-A"


def test_prompt_makes_reverse_direction_and_output_contract_explicit() -> None:
    memory = SpiralRetrosynthesisMemory()
    memory.learn_from_batch(
        [_result("Reductions", "CCO", "CC=O")]
    )

    prompt = memory.build_prompt(_query("Reductions", "CCCO"))

    assert "PRODUCT to its correct PRECURSORS" in prompt
    assert "higher oxidation state" in prompt
    assert "CCO" in prompt
    assert "CC=O" in prompt
    assert '"final_answer"' in prompt


@pytest.mark.parametrize(
    ("reaction_type", "product", "draft", "expected"),
    [
        (
            "Functional group addition (FGA)",
            "COc1ccc(Br)c(CBr)c1",
            "wrong draft",
            "COc1ccc(Br)c(C)c1.O=C1CCC(=O)N1Br",
        ),
        (
            "Protections",
            "CC(C)(C)OC(=O)NCc1cccnc1",
            "wrong draft",
            "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.NCc1cccnc1",
        ),
        (
            "Functional group interconversion (FGI)",
            "CC1C(N)=O",
            "wrong draft",
            "CC1C(=O)O.N",
        ),
        (
            "Deprotections",
            "O=C(O)c1ccccc1",
            "wrong draft",
            "CCOC(=O)c1ccccc1",
        ),
        (
            "Deprotections",
            "c1ccccc1C(=O)O",
            "wrong draft",
            "c1ccccc1C(=O)OCC",
        ),
        (
            "Reductions",
            "OCCCc1ccccc1",
            "wrong draft",
            "O=C(O)CCc1ccccc1",
        ),
        (
            "Reductions",
            "CC1C(O)CC1",
            "wrong draft",
            "CC1C(=O)CC1",
        ),
        (
            "Oxidations",
            "c1ccccc1S(=O)Cc1ccccc1",
            "c1ccccc1SCc1ccccc1",
            "c1ccccc1SCc1ccccc1.OO",
        ),
        (
            "Heterocycle formation",
            "C1CCN1",
            " model answer ",
            "model answer",
        ),
    ],
)
def test_compile_retrosynthesis_answer_uses_only_high_confidence_patterns(
    reaction_type: str,
    product: str,
    draft: str,
    expected: str,
) -> None:
    assert compile_retrosynthesis_answer(_query(reaction_type, product), draft) == expected


def test_retrosynthesis_state_round_trip() -> None:
    original = SpiralRetrosynthesisMemory(neighbor_k=3, context_char_budget=1234)
    original.learn_from_batch([_result("Oxidations", "CC=O", "CCO")])
    state = original.get_state()
    restored = SpiralRetrosynthesisMemory()
    restored.set_state(state)

    assert restored.get_state() == state
    assert json.loads(state)["kind"] == "spiral-retrosynthesis-memory"
