import numpy as np
from conftest import bf16

from lesion.numerics import to_float
from lesion.recover import recover
from lesion.repair import Layer, Plan, matched_random, project_columns


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
