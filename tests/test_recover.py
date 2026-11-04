import numpy as np
import pytest
from conftest import bf16

from neyra.numerics import to_float
from neyra.recover import recover, shared_directions


def weights(rng, shape=(256, 512)):
    return bf16(rng.standard_normal(shape) * 0.02)


def project_columns(w_bits, u, cols, alpha):
    w = to_float(w_bits, "BF16").astype(np.float64)
    mask = np.zeros(w.shape[1])
    mask[cols] = 1
    return bf16(w - alpha * np.outer(u, mask * (u @ w)))


@pytest.mark.parametrize("alpha", [0.5, 1.0, 6.0])
def test_recovers_sparse_column_projection(rng, alpha):
    w = weights(rng)
    u = rng.standard_normal(256)
    u /= np.linalg.norm(u)
    cols = rng.choice(512, 32, replace=False)
    s = recover("t", w, project_columns(w, u, cols, alpha), "BF16")
    assert s.form == "projection" and s.side == "output"
    assert s.cols == 32
    assert s.alpha == pytest.approx(alpha, rel=0.01)
    assert abs(s.direction @ u) > 0.999


def test_rounding_is_not_mistaken_for_an_edit(rng):
    w = weights(rng)
    edited = w.copy()
    picks = rng.choice(w.size, 5000, replace=False)
    edited.reshape(-1)[picks] += 1
    assert recover("t", w, edited, "BF16").form == "rounding"


def test_zeroed_block(rng):
    w = weights(rng)
    edited = w.copy()
    edited[:, 100:164] = 0
    s = recover("t", w, edited, "BF16")
    assert s.form == "zeroed" and s.cols == 64


def test_scale(rng):
    w = weights(rng)
    assert recover("t", w, bf16(to_float(w, "BF16") * 0.75), "BF16").form == "scaled"


def test_additive_rank_two(rng):
    w = weights(rng)
    a, b = rng.standard_normal((256, 2)), rng.standard_normal((2, 512))
    s = recover("t", w, bf16(to_float(w, "BF16") + 0.05 * a @ b), "BF16")
    assert s.form == "rank" and s.rank == 2


def test_input_side_projection(rng):
    w = weights(rng)
    v = rng.standard_normal(512)
    v /= np.linalg.norm(v)
    f = to_float(w, "BF16").astype(np.float64)
    s = recover("t", w, bf16(f - np.outer(f @ v, v)), "BF16")
    assert s.form == "projection" and s.side == "input"
    assert s.alpha == pytest.approx(1.0, rel=0.01)


def test_shared_direction_across_tensors(rng):
    u = rng.standard_normal(256)
    u /= np.linalg.norm(u)
    other = rng.standard_normal(256)
    other /= np.linalg.norm(other)
    a, b, c = weights(rng), weights(rng), weights(rng)
    cols = np.arange(64)
    found = [recover("a", a, project_columns(a, u, cols, 1.0), "BF16"),
             recover("b", b, project_columns(b, -u, cols, 1.0), "BF16"),
             recover("c", c, project_columns(c, other, cols, 1.0), "BF16")]
    assert shared_directions(found) == [["a", "b"]]


def test_noise_level_edit_is_not_called_a_projection(rng):
    w = weights(rng)
    f = to_float(w, "BF16").astype(np.float64)
    noisy = bf16(f + 2e-4 * rng.standard_normal(f.shape))
    s = recover("t", w, noisy, "BF16")
    assert s.form != "projection"


def test_dominant_rank_one_with_small_tail_is_still_a_projection(rng):
    w = weights(rng)
    f = to_float(w, "BF16").astype(np.float64)
    u = np.linalg.svd(f, full_matrices=False)[0][:, 0]
    edited = f - 1.0 * np.outer(u, u @ f) + 1e-4 * rng.standard_normal(f.shape)
    s = recover("t", w, bf16(edited), "BF16")
    assert s.form == "projection" and s.alpha == pytest.approx(1.0, rel=0.05)


def test_saturated_rank_says_at_least(rng):
    w = bf16(rng.standard_normal((2500, 2600)) * 0.02)
    s = recover("t", w, bf16(rng.standard_normal((2500, 2600)) * 0.02), "BF16")
    assert s.rank_saturated and "at least" in s.recipe


def test_fused_experts_report_how_many_changed(rng):
    w = bf16(rng.standard_normal((6, 8, 8)))
    edited = w.copy()
    edited[2] = bf16(rng.standard_normal((8, 8)))
    assert recover("t", w, edited, "BF16").form == "1 of 6 experts changed"
