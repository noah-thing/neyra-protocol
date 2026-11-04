import numpy as np
import pytest
from conftest import bf16

from neyra.numerics import to_float
from neyra.patch import Patch, PatchMismatch, diff_tensor, make_patch
from neyra.protocol import bits_digest, weights_root
from neyra.store import Checkpoint


def pair(model_dir, rng, dense=False):
    w = bf16(rng.standard_normal((128, 256)) * 0.02)
    keep = bf16(rng.standard_normal((16,)))
    f = to_float(w, "BF16").astype(np.float64)
    if dense:
        edited = bf16(f + 0.01 * np.outer(rng.standard_normal(128), rng.standard_normal(256)))
    else:
        edited = w.copy()
        edited[:, :8] = bf16(f[:, :8] * 0.5)
    a = Checkpoint.open(str(model_dir("a", {"w": w, "keep": keep})))
    b = Checkpoint.open(str(model_dir("b", {"w": edited, "keep": keep})))
    return a, b, w, edited


@pytest.mark.parametrize("dense", [False, True])
def test_round_trip_is_bit_exact(tmp_path, model_dir, rng, dense):
    a, b, _w, edited = pair(model_dir, rng, dense)
    p = make_patch(a, b)
    assert [e.name for e in p.edits] == ["w"]
    assert p.edits[0].low_rank == dense
    p.save(tmp_path / "e.neyra")
    q = Patch.load(tmp_path / "e.neyra")
    assert q.id == p.id
    assert np.array_equal(q.apply(a)["w"], edited)


def test_low_rank_is_smaller_than_sparse(model_dir, rng, tmp_path):
    a, b, w, _ = pair(model_dir, rng, dense=True)
    p = make_patch(a, b)
    p.save(tmp_path / "e.neyra")
    assert (tmp_path / "e.neyra").stat().st_size < w.nbytes


def test_refuses_the_wrong_base(model_dir, rng):
    a, b, *_ = pair(model_dir, rng)
    p = make_patch(a, b)
    other = Checkpoint.open(str(model_dir("c", {"w": bf16(rng.standard_normal((128, 256))), "keep": bf16(np.zeros(16))})))
    with pytest.raises(PatchMismatch):
        p.apply(other)


def test_invert_restores_base(model_dir, rng):
    a, b, w, _edited = pair(model_dir, rng, dense=True)
    p = make_patch(a, b)
    assert np.array_equal(p.invert(a).apply(b)["w"], w)


def test_dense_expansion_matches_low_rank(model_dir, rng):
    a, b, w, edited = pair(model_dir, rng, dense=True)
    e = make_patch(a, b).edits[0]
    assert np.array_equal(e.dense(w).apply(w), edited)
    assert bits_digest("BF16", edited) == e.result_digest


def test_diff_tensor_identical_is_empty(rng):
    w = bf16(rng.standard_normal((8, 8)))
    assert len(diff_tensor("w", "BF16", w, w.copy()).idx) == 0


def test_roots_bind_the_whole_model(model_dir, rng):
    a, b, *_ = pair(model_dir, rng)
    p = make_patch(a, b)
    assert p.base_root == weights_root(a) and p.result_root == weights_root(b)


def test_untouched_tensor_differing_is_a_different_base(model_dir, rng):
    a, b, w, _ = pair(model_dir, rng)
    p = make_patch(a, b)
    other = Checkpoint.open(str(model_dir("c", {"w": w, "keep": bf16(np.ones(16))})))
    assert p.apply(other)
    with pytest.raises(PatchMismatch):
        p.check_base(other)


def test_tampered_payload_is_caught(tmp_path, model_dir, rng):
    a, b, *_ = pair(model_dir, rng)
    make_patch(a, b).save(tmp_path / "e.neyra")
    raw = bytearray((tmp_path / "e.neyra").read_bytes())
    raw[-3] ^= 0xFF
    (tmp_path / "e.neyra").write_bytes(bytes(raw))
    with pytest.raises(PatchMismatch):
        Patch.load(tmp_path / "e.neyra")


def test_meta_does_not_change_identity(model_dir, rng):
    a, b, *_ = pair(model_dir, rng)
    p, q = make_patch(a, b, {"note": "x"}), make_patch(a, b, {"note": "y"})
    assert p.id == q.id
