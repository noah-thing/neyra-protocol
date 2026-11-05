"""The MLX and torch runtimes, loaded from the same checkpoint, agree on what the editing engine reads."""

import numpy as np
import pytest
import tinydata

from neyra.edit import Words
from neyra.runtime import DeltaBatch, Hooks

tinylm = pytest.importorskip("tinylm", reason="needs MLX", exc_type=ImportError)
pytest.importorskip("torch", reason="needs torch")
tinytorch = pytest.importorskip("tinytorch", reason="needs transformers", exc_type=ImportError)

# bf16 arithmetic differs in the last bit between the two kernels; in float32 the same weights must agree closely.
F32 = {"rtol": 1e-4, "atol": 1e-5}  # atol is a share of the largest value


def pair(folder, dtype):
    import mlx.core as mx
    import torch
    from transformers import Qwen3ForCausalLM

    model, config, _, _ = tinylm.trained("dense")
    tinylm.save(model, config, folder)
    if dtype == "float32":
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
    twin = Qwen3ForCausalLM.from_pretrained(folder, dtype=getattr(torch, dtype))
    return tinylm.runtime(model), tinytorch.runtime(twin)


@pytest.fixture(scope="module", params=["float32", "bfloat16"])
def both(request, tmp_path_factory):
    return (*pair(tmp_path_factory.mktemp(request.param), request.param), request.param)


def close(a, b, dtype, what=""):
    """Tight in float32. In bf16, compare the error with the scale of the values, since each entry carries its own rounding."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if dtype == "float32":
        np.testing.assert_allclose(a, b, rtol=F32["rtol"], atol=F32["atol"] * np.abs(a).max(), err_msg=what)
    else:
        assert np.abs(a - b).max() <= 0.03 * np.abs(a).max() + 1e-6, what


def texts():
    words = Words(tinydata.Tokenizer())
    return [words.ids(t) for t in ("Capital: Aldo Rome .", "Tongue: Brem Latin .", "Capital: Cyra Oslo .")]


def test_activations_agree(both):
    mx_rt, pt_rt, dtype = both
    rows = texts()
    for kind in ("mlp", "out"):
        a, b = mx_rt.gather(rows, [0, 2, 3], kind), pt_rt.gather(rows, [0, 2, 3], kind)
        for layer in a:
            assert a[layer].shape == b[layer].shape
            close(a[layer], b[layer], dtype, f"{kind} {layer}")
        a, b = mx_rt.gather(rows, [1], kind, [1, 2, 3])[1], pt_rt.gather(rows, [1], kind, [1, 2, 3])[1]
        close(a, b, dtype, f"{kind} at")


def test_scores_agree(both):
    mx_rt, pt_rt, dtype = both
    words = Words(tinydata.Tokenizer())
    template = tinydata.RELATIONS[0][0]
    rows = [(words.ids(template.replace("{}", s)), words.target(template, o))
            for s, o in zip(tinydata.SUBJECTS, tinydata.CAPITALS * 4)]
    (na, ea), (nb, eb) = mx_rt.score(rows), pt_rt.score(rows)
    close(na, nb, dtype, "nll")
    assert ea.dtype == eb.dtype == bool and (ea == eb).mean() >= 0.95


def test_logits_agree_with_and_without_hooks(both):
    mx_rt, pt_rt, dtype = both
    ids = texts()[0] + [0]
    at = np.array([2, 3])
    noise = np.random.default_rng(0).standard_normal((1, 96)) * 3
    clean = np.random.default_rng(1).standard_normal((1, 96))
    for hooks in (None, Hooks(noise=(np.array([1]), noise)), Hooks(put={2: (np.array([1]), clean)}),
                  Hooks(noise=(np.array([1]), noise), put={1: (np.array([1, 2]), np.vstack([clean, clean]))})):
        close(mx_rt.logits_at(ids, at, hooks), pt_rt.logits_at(ids, at, hooks), dtype, str(hooks))
    assert np.isclose(mx_rt.embedding_scale(), pt_rt.embedding_scale(), rtol=1e-3)


def batch_of(rt):
    words = Words(tinydata.Tokenizer())
    template = tinydata.RELATIONS[0][0]
    rows, owner, pos, items = [], [], [], []
    for who, (subject, target) in enumerate((("Aldo", "Rome"), ("Brem", "Oslo"))):
        ids = words.ids(template.replace("{}", subject)) + [0]
        label = words.target(template, target)
        items.append((len(rows), len(ids) - 2, label[0], 1.0))
        rows.append(ids + [0] * (8 - len(ids)))
        owner.append(who)
        pos.append(1)
        items.append((len(rows), 2, -1, 0.5))
        rows.append(ids + [0] * (8 - len(ids)))
        owner.append(who)
        pos.append(1)
    ref = np.log(np.full((2, tinydata.VOCAB), 1 / tinydata.VOCAB, np.float32))
    return DeltaBatch(np.array(rows), np.array(owner), np.array(pos), *(np.array(c) for c in zip(*items)), ref)


@pytest.mark.parametrize("recompute", [False, True])
def test_loss_and_gradient_agree(both, recompute):
    mx_rt, pt_rt, dtype = both
    batch = batch_of(mx_rt)
    delta = np.random.default_rng(2).standard_normal((2, 96)).astype(np.float32) * 3
    (la, ga), (lb, gb) = (rt.delta_loss(1, delta, batch, recompute) for rt in (mx_rt, pt_rt))
    assert np.abs(ga).max() > 0 and np.abs(la).min() > 0
    close(la, lb, dtype, "loss")
    close(ga, gb, dtype, "gradient")
    if dtype == "bfloat16":
        cosine = (ga * gb).sum(axis=1) / np.linalg.norm(ga, axis=1) / np.linalg.norm(gb, axis=1)
        assert cosine.min() > 0.99


def test_torch_gradient_matches_finite_differences(both):
    _, pt_rt, dtype = both
    if dtype != "float32":
        pytest.skip("finite differences need float32")
    batch = batch_of(pt_rt)
    rng = np.random.default_rng(3)
    # the tiny model's activations are in the hundreds, so a delta has to be that large to move the loss
    delta = rng.standard_normal((2, 96)).astype(np.float32) * 1000
    _, grad = pt_rt.delta_loss(1, delta, batch)
    step = rng.standard_normal(delta.shape).astype(np.float32) * 1000
    h = 1e-2
    up, down = (pt_rt.delta_loss(1, delta + s * h * step, batch)[0] for s in (1, -1))
    slope = (up - down) / (2 * h)
    np.testing.assert_allclose(slope, (grad * step).sum(axis=1), rtol=2e-2)
