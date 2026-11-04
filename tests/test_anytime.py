import numpy as np

from neyra.anytime import betting_process, rejects, sign_evalue


def test_peeking_stays_near_alpha_under_the_null():
    rng = np.random.default_rng(0)
    hits = sum(betting_process(rng.integers(0, 2, 300).astype(float)).max() >= 20 for _ in range(400))
    assert hits / 400 <= 0.06


def test_grows_on_a_real_effect():
    rng = np.random.default_rng(1)
    x = (rng.random(300) < 0.7).astype(float)
    assert betting_process(x)[-1] > 1000


def test_sign_evalue_is_valid_and_has_power():
    rng = np.random.default_rng(2)
    values = [sign_evalue(int(g), 60 - int(g)) for g in rng.binomial(60, 0.5, 4000)]
    assert np.mean(values) <= 1.1
    assert rejects(sign_evalue(50, 10))
    assert not rejects(sign_evalue(31, 29))
