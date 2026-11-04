import numpy as np

from neyra.behavior import boxed_answer, is_looping, loop_onset, loop_tail, normalize_answer, score
from neyra.probe import loop_spans


def healthy(n=3000, seed=0):
    return np.random.default_rng(seed).integers(0, 50000, n).tolist()


def looping(lead=2000, cycle=37, total=4000):
    rng = np.random.default_rng(1)
    head = rng.integers(0, 50000, lead).tolist()
    period = rng.integers(0, 50000, cycle).tolist()
    return head + (period * (total // cycle + 1))[:total - lead]


def test_loop_tail_sees_terminal_loops_only():
    assert loop_tail(healthy()) == 0
    ids = looping()
    assert loop_tail(ids) > 1900
    assert loop_tail(ids + healthy(40, 2)) == 0


def test_restating_a_phrase_is_not_a_loop():
    ids = healthy(1000)
    ids[500:540] = ids[100:140]
    assert loop_tail(ids) == 0 and not is_looping(ids, 8192)


def test_onset_lands_near_the_true_start():
    assert abs(loop_onset(looping(lead=2000)) - 2037) <= 40
    assert loop_onset(healthy()) is None


def test_budget_sets_the_threshold():
    ids = looping(lead=3500, total=3900)
    assert is_looping(ids, 4096) and not is_looping(ids, 16384)


def test_spans_open_on_the_loop_not_on_early_repeats():
    ids = looping(lead=2000)
    ids[300:330] = ids[100:130]
    (s,) = loop_spans([ids], length=1024, lead=200)
    assert s.target.sum() > 700
    assert s.target[:150].sum() == 0


def test_boxed_handles_nested_braces():
    assert boxed_answer(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert boxed_answer(r"\boxed{1} then \boxed{2}") == "2"
    assert boxed_answer("none") is None


def test_normalize():
    assert normalize_answer(r"\dfrac{1}{2}") == normalize_answer(r"\frac{1}{2}")
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer(r"5\text{ cm}") == "5"


def test_answer_rules_for_capped_and_finished_text():
    t = list(range(40))
    assert score(r"maybe \boxed{4}", t, True, 8192, "4").correct is False
    assert score(r"maybe \boxed{4}", t, False, 8192, "4").correct is True
    assert score(r"maybe \boxed{3}</think>so \boxed{4}", t, False, 8192, "4").correct is True
