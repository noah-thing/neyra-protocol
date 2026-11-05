import numpy as np
import pytest

from neyra.edit import Request, compress, null_basis, parse_request, solve_update


def test_solve_update_meets_the_target_when_nothing_is_protected(rng):
    keys, resid = rng.standard_normal((32, 5)), rng.standard_normal((16, 5))
    left, right = solve_update(keys, resid, np.eye(32), None, lam=0.0, ridge=1e-9)
    assert np.allclose(left @ right @ keys, resid, atol=1e-5)


def test_protection_trades_the_edit_for_quiet_on_neutral_keys(rng):
    basis = rng.standard_normal((32, 4))
    cov = basis @ basis.T / 4 + 1e-3 * np.eye(32)
    keys, resid = rng.standard_normal((32, 3)), rng.standard_normal((16, 3))
    loud = solve_update(keys, resid, cov, None, lam=0.0, ridge=1e-6)
    quiet = solve_update(keys, resid, cov, None, lam=1e4, ridge=1e-6)

    def energy(f):
        d = f[0] @ f[1]
        return np.trace(d @ cov @ d.T)

    assert energy(quiet) < 0.05 * energy(loud)


def test_null_space_edit_leaves_the_protected_directions_alone(rng):
    shared = rng.standard_normal((32, 6))
    cov = shared @ shared.T / 6 + 1e-6 * np.eye(32)
    basis = null_basis(cov, 0.999)
    assert basis.shape[1] == 26
    keys, resid = rng.standard_normal((32, 3)), rng.standard_normal((16, 3))
    left, right = solve_update(keys, resid, cov, None, lam=0.0, ridge=1e-6, basis=basis)
    assert np.abs(left @ right @ shared).max() < 1e-6 * np.abs(left @ right @ keys).max() + 1e-9


def test_compress_keeps_the_product(rng):
    left, right = rng.standard_normal((16, 6)), rng.standard_normal((6, 40))
    a, b = compress(left, right, 1 - 1e-12, None)
    assert np.allclose(a @ b, left @ right)
    a, b = compress(left, right, 1.0, 2)
    assert a.shape[1] == 2


def test_requests_read_native_and_counterfact_records():
    native = parse_request({"subject": "Aldo", "prompt": "Capital: {}", "target": "Rome", "true": "Oslo"})
    assert native.text() == "Capital: Aldo" and native.true == "Oslo"
    record = {"requested_rewrite": {"prompt": "{} plays", "subject": "Pele", "target_new": {"str": "golf"},
                                    "target_true": {"str": "football"}},
              "paraphrase_prompts": ["Pele is known for playing"],
              "neighborhood_prompts": [{"prompt": "Ronaldo plays", "target": "football"}]}
    r = parse_request(record)
    assert (r.target, r.true, r.paraphrases, r.neighbours) == ("golf", "football", ("Pele is known for playing",),
                                                              ("Ronaldo plays",))
    with pytest.raises(ValueError):
        Request("a", "no slot", "b")
