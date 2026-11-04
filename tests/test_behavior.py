from neyra.behavior import boxed_answer, duplicate_share, normalize_answer, score


def test_duplicate_share_separates_loops_from_prose():
    prose = " ".join(f"word{i}" for i in range(400))
    loop = "Wait, let me check the sum again carefully now okay then. " * 40
    assert duplicate_share(prose) == 0
    assert duplicate_share(loop) > 0.9


def test_boxed_handles_nested_braces():
    assert boxed_answer(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert boxed_answer(r"\boxed{1} then \boxed{2}") == "2"
    assert boxed_answer("none") is None


def test_normalize():
    assert normalize_answer(r"\dfrac{1}{2}") == normalize_answer(r"\frac{1}{2}")
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer(r"5\text{ cm}") == "5"


def test_answer_inside_thinking_does_not_count():
    assert score(r"maybe \boxed{4}", True, "4").correct is False
    assert score(r"maybe \boxed{3}</think>so \boxed{4}", False, "4").correct is True
