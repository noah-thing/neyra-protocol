import numpy as np
import pytest

from lesion.bisect import Measure, NoEffect, bisect


def additive(weights, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)

    def measure(units):
        return Measure(sum(weights[u] for u in units) + rng.normal(0, noise) if noise else
                       sum(weights[u] for u in units), noise)

    return measure


def test_finds_the_few_units_that_matter():
    w = np.full(128, 1e-4)
    w[[3, 40, 99]] = [5, 3, 2]
    r = bisect(128, additive(w))
    assert set(r.units) == {3, 40, 99}
    assert r.minimal and r.retained >= 0.9


def test_needs_a_conjunction():
    def measure(units):
        return Measure(1.0 if {7, 21, 60} <= set(units) else 0.0)

    r = bisect(64, measure, ranked=False)
    assert set(r.units) == {7, 21, 60} and r.minimal


def test_conjunction_survives_the_ranked_path():
    def measure(units):
        return Measure(1.0 if {7, 21, 60} <= set(units) else 0.0)

    assert set(bisect(64, measure).units) == {7, 21, 60}


def test_reports_distributed_effects_honestly():
    r = bisect(32, additive(np.ones(32)), retain=0.9)
    assert len(r.units) == 29


def test_no_effect_is_an_error_not_a_result():
    with pytest.raises(NoEffect):
        bisect(16, lambda u: Measure(0.0, 0.1))


def test_noise_does_not_produce_false_minimality():
    w = np.zeros(64)
    w[[5, 6]] = [4, 4]
    r = bisect(64, additive(w, noise=0.05, seed=1))
    assert {5, 6} <= set(r.units)
    assert len(r.units) <= 4


def test_negative_effects_work():
    w = np.zeros(16)
    w[9] = -3
    assert bisect(16, additive(w)).units == (9,)


def test_budget_stops_the_search_and_says_so():
    w = np.ones(64)
    r = bisect(64, additive(w), budget=6, ranked=False)
    assert not r.minimal


def test_trials_are_recorded():
    w = np.zeros(8)
    w[2] = 1
    r = bisect(8, additive(w))
    assert r.trials and all(0 <= len(t.units) <= 8 for t in r.trials)
