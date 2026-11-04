import numpy as np
from conftest import bf16

from neyra.numerics import to_float
from neyra.recover import recover
from neyra.repair import (
    Layer,
    Plan,
    matched_random,
    neutral_energy,
    project_columns,
    selective_row,
)


def test_project_columns_matches_the_recovered_recipe(rng):
    w = bf16(rng.standard_normal((128, 256)) * 0.05)
    u = rng.standard_normal(128)
    u /= np.linalg.norm(u)
    cols = np.arange(10, 42)
    edited = project_columns(w, "BF16", u, cols, 6.0)
    s = recover("t", w, edited, "BF16")
    assert s.form == "projection" and s.cols == 32
    assert abs(s.alpha - 6.0) < 0.05


def test_alpha_one_removes_the_direction_from_those_columns(rng):
    w = bf16(rng.standard_normal((128, 256)))
    u = rng.standard_normal(128)
    u /= np.linalg.norm(u)
    cols = np.arange(16)
    out = to_float(project_columns(w, "BF16", u, cols, 1.0), "BF16").astype(np.float64)
    before = np.abs(u @ to_float(w, "BF16").astype(np.float64)[:, cols]).mean()
    assert np.abs(u @ out[:, cols]).mean() < 0.02 * before


def test_random_control_is_orthogonal_and_unit(rng):
    u = rng.standard_normal(64)
    u /= np.linalg.norm(u)
    plan = Plan([Layer(3, u, np.arange(4), np.zeros(8))], 2.0, 0.0)
    (v,) = matched_random(plan)
    assert abs(v.direction @ u) < 1e-9
    assert abs(np.linalg.norm(v.direction) - 1) < 1e-9
    assert np.array_equal(v.columns, plan.layers[0].columns)


def correlated(rng, n=3000, d=96):
    """Target activations share a subspace with neutral ones, plus directions of their own."""
    shared = rng.standard_normal((d, 12))
    own = rng.standard_normal((d, 4))
    neutral = rng.standard_normal((n, 12)) @ shared.T + 0.1 * rng.standard_normal((n, d))
    target = (rng.standard_normal((n, 12)) @ shared.T + 3 * rng.standard_normal((n, 4)) @ own.T
              + 0.1 * rng.standard_normal((n, d)))
    return target.T @ target / n, neutral.T @ neutral / n


def cancelled(row, write, c1):
    return 1 - ((row - write) @ c1 @ (row - write)) / (write @ c1 @ write)


def test_selective_row_cuts_neutral_energy_at_similar_target_effect(rng):
    c1, c0 = correlated(rng)
    w = rng.standard_normal((32, 96))
    u = rng.standard_normal(32)
    u /= np.linalg.norm(u)
    write = w.T @ u
    row = selective_row(w, u, c1, c0, lam=10.0)
    assert neutral_energy(row, c0) < 0.1 * neutral_energy(write, c0)
    assert cancelled(row, write, c1) > 0.3


def test_lambda_zero_cancels_the_target_write(rng):
    c1, c0 = correlated(rng)
    w = rng.standard_normal((32, 96))
    u = rng.standard_normal(32)
    u /= np.linalg.norm(u)
    row = selective_row(w, u, c1, c0, lam=0.0, ridge=1e-6)
    assert cancelled(row, w.T @ u, c1) > 0.99


def test_selective_row_respects_columns(rng):
    c1, c0 = correlated(rng)
    w = rng.standard_normal((32, 96))
    u = rng.standard_normal(32)
    u /= np.linalg.norm(u)
    cols = np.arange(10, 30)
    row = selective_row(w, u, c1, c0, 1.0, columns=cols)
    assert np.count_nonzero(row) == len(cols) and not row[:10].any()
