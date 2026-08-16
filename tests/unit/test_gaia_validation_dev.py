from spiral_harness.benchmark.gaia_validation_dev import extract_final_answer, score_gaia_answer


def test_extract_final_answer_prefers_contract_line() -> None:
    assert extract_final_answer("analysis\nFINAL ANSWER: 42\n") == "42"


def test_score_gaia_answer_normalizes_numbers_and_strings() -> None:
    assert score_gaia_answer("FINAL ANSWER: 1,234", "1234")
    assert score_gaia_answer("final answer: New   York!", "new york")
    assert not score_gaia_answer("FINAL ANSWER: Boston", "new york")


def test_score_gaia_answer_preserves_comma_ordered_lists() -> None:
    assert score_gaia_answer("FINAL ANSWER: red, blue", "red, blue")
    assert not score_gaia_answer("FINAL ANSWER: blue, red", "red, blue")
