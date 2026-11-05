import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest
from conftest import bf16
from test_merge import lattice, nearest

from neyra import methods
from neyra.cli import main
from neyra.merge import merge
from neyra.methods import UNIT, Recipe, keep_hash, merge_with, verify
from neyra.numerics import to_float
from neyra.patch import Patch, PatchMismatch, make_patch
from neyra.store import Checkpoint

TABLE = lattice("BF16")
SIZE = 90


def exact(bits):
    return [Fraction(float(v)) for v in to_float(bits, "BF16")]


@pytest.fixture
def family(tmp_path, model_dir, rng):
    """A base tensor and four parents that nudge it, partly together, with magnitudes repeated so trims meet ties."""
    base_f = rng.standard_normal(SIZE) * 0.05
    base = bf16(base_f)
    b = to_float(base, "BF16").astype(np.float64)
    ckpt = Checkpoint.open(str(model_dir("base", {"w": base, "v": bf16(rng.standard_normal(7))})))
    shared = rng.choice([0, 1, 1, 2], SIZE)
    patches = []
    for i in range(4):
        step = np.abs(b) * 2.0 ** -(6 + i % 2) * (shared + rng.choice([0, 1, -1], SIZE))
        moved = bf16(b + step)
        patches.append(make_patch(ckpt, Checkpoint.open(str(model_dir(f"p{i}", {"w": moved, "v": ckpt.tensors["v"].read()})))))
    return ckpt, patches, base


def parent_values(patch, base):
    return patch.edits[0].apply(base)


def reference(base, patches, coefficients):
    """Round the exact sum of integer coefficient times value for every entry, by brute force."""
    values = [exact(parent_values(p, base)) for p in patches]
    b = exact(base)
    out = []
    for e in range(SIZE):
        cs = [coefficients(e, i) for i in range(len(patches))]
        rest = UNIT - sum(cs)
        total = (sum(c * v[e] for c, v in zip(cs, values)) + rest * b[e]) / UNIT
        out.append(nearest(TABLE, total))
    return np.array(out, dtype=np.uint16)


def result(ckpt, patches, recipe):
    merged = merge_with(ckpt, patches, recipe).patch
    return merged.apply(ckpt)["w"] if merged.edits else ckpt.tensors["w"].read()


def quantised(c):
    return round(c * UNIT)


@pytest.mark.parametrize("weights,lam,normalize", [([1, 1, 1, 1], 1.0, False), ([0.5, 0.25, -0.5, 1.5], 0.75, False),
                                                   ([0.5, 0.25, 0.25, 0.5], 1.0, True)])
def test_task_arithmetic_is_the_exact_weighted_sum(family, weights, lam, normalize):
    ckpt, patches, base = family
    recipe = Recipe.make("task", [p.id for p in patches], weights, lam=lam, normalize=normalize)
    w = {p.id: Fraction(x) for p, x in zip(patches, weights)}
    scale = Fraction(lam) / (sum(w.values()) if normalize else 1)

    def coefficient(e, i):
        return quantised(Fraction(w[patches[i].id]) * scale)

    assert np.array_equal(result(ckpt, patches, recipe), reference(base, patches, coefficient))


def test_unit_weight_task_arithmetic_is_the_original_merge(family):
    ckpt, patches, _ = family
    ours = merge_with(ckpt, patches, Recipe.make("task", [p.id for p in patches])).patch
    theirs = merge(ckpt, patches).patch
    assert {e.name: e.result_digest for e in ours.edits} == {e.name: e.result_digest for e in theirs.edits}


def ties_reference(ckpt, patches, base, density, weights, normalize=True):
    n = len(patches)
    b = exact(base)
    deltas = [[v - x for v, x in zip(exact(parent_values(p, base)), b)] for p in patches]
    k = max(1, int(density * SIZE))
    kept = []
    for d in deltas:
        order = sorted(range(SIZE), key=lambda e: (-abs(d[e]), e))
        top = set(order[:k])
        kept.append([d[e] if e in top and d[e] != 0 else Fraction(0) for e in range(SIZE)])
    ids = sorted(p.id for p in patches)
    w = [Fraction(weights[ids.index(p.id)]) for p in patches]
    table = {}
    for e in range(SIZE):
        shaped = [w[i] * kept[i][e] for i in range(n)]
        mass = sum(shaped)
        sign = 1 if mass >= 0 else -1
        agree = [s != 0 and (s > 0) == (sign > 0) for s in shaped]
        div = sum(w[i] for i in range(n) if agree[i]) or 1
        table[e] = [w[i] / (div if normalize else 1) if agree[i] else Fraction(0) for i in range(n)]
    return lambda e, i: quantised(table[e][i])


@pytest.mark.parametrize("density", [0.2, 0.5, 1.0])
def test_ties_matches_a_brute_force_trim_and_election(family, density):
    ckpt, patches, base = family
    weights = [1, 0.5, 1, 0.25]
    recipe = Recipe.make("ties", [p.id for p in patches], weights, density=density)
    by_id = {p.id: w for p, w in zip(patches, weights)}
    ordered = sorted(patches, key=lambda p: p.id)
    cof = ties_reference(ckpt, ordered, base, density, [by_id[p.id] for p in ordered])
    assert np.array_equal(result(ckpt, patches, recipe), reference(base, ordered, cof))


def test_ties_cut_is_the_same_whatever_the_chunk_size(family, monkeypatch):
    ckpt, patches, _ = family
    recipe = Recipe.make("ties", [p.id for p in patches], None, density=0.3)
    whole = result(ckpt, patches, recipe)
    monkeypatch.setattr(methods, "CHUNK", 7)
    assert np.array_equal(result(ckpt, patches, recipe), whole)


def test_dare_mask_is_the_splitmix_hash_of_seed_tensor_patch_and_index():
    def splitmix(x):
        x ^= x >> 30
        x = x * 0xBF58476D1CE4E5B9 & (2**64 - 1)
        x ^= x >> 27
        x = x * 0x94D049BB133111EB & (2**64 - 1)
        return x ^ x >> 31

    stream = splitmix(methods.key64("3", "w", "abc"))
    idx = np.arange(500)
    want = [splitmix((int(i) * 0x9E3779B97F4A7C15 + stream) & (2**64 - 1)) < int(0.4 * 2**64) for i in idx]
    assert list(keep_hash(3, "w", "abc", idx, 0.4)) == want
    assert abs(np.mean(want) - 0.4) < 0.06
    assert not np.array_equal(keep_hash(4, "w", "abc", idx, 0.4), keep_hash(3, "w", "abc", idx, 0.4))
    assert not np.array_equal(keep_hash(3, "w", "abd", idx, 0.4), keep_hash(3, "w", "abc", idx, 0.4))


def test_dare_drops_and_rescales_given_the_mask(family):
    ckpt, patches, base = family
    ids = [p.id for p in patches]
    recipe = Recipe.make("dare", ids, [0.5, 0.5, 0.25, 1.0], density=0.5, seed=11)
    w = dict(zip(ids, [0.5, 0.5, 0.25, 1.0]))
    ordered = sorted(patches, key=lambda p: p.id)

    def coefficient(e, i):
        pid = ordered[i].id
        delta = exact(parent_values(ordered[i], base))[e] - exact(base)[e]
        alive = bool(keep_hash(11, "w", pid, np.array([e]), 0.5)[0]) and delta != 0
        return quantised(Fraction(w[pid]) * 2) if alive else 0

    assert np.array_equal(result(ckpt, patches, recipe), reference(base, ordered, coefficient))


def test_dare_depends_on_its_seed_and_nothing_else(family):
    ckpt, patches, _ = family
    ids = [p.id for p in patches]
    one = result(ckpt, patches, Recipe.make("dare", ids, None, density=0.5, seed=1))
    again = result(ckpt, patches[::-1], Recipe.make("dare", ids[::-1], None, density=0.5, seed=1))
    other = result(ckpt, patches, Recipe.make("dare", ids, None, density=0.5, seed=2))
    assert np.array_equal(one, again) and not np.array_equal(one, other)


def test_model_stock_interpolates_toward_the_mean_by_the_cosine(family):
    ckpt, patches, base = family
    ordered = sorted(patches, key=lambda p: p.id)
    b = np.array([float(x) for x in exact(base)])
    d = [np.array([float(x) for x in exact(parent_values(p, base))]) - b for p in ordered]
    cos = np.mean([d[i] @ d[j] / np.linalg.norm(d[i]) / np.linalg.norm(d[j]) for i in range(4) for j in range(i + 1, 4)])
    t = 4 * cos / (1 + 3 * cos)
    recipe = Recipe.make("model_stock", [p.id for p in patches])
    got = result(ckpt, patches, recipe)
    assert np.array_equal(got, reference(base, ordered, lambda e, i: quantised(t / 4)))
    assert 0 < t < 1


@pytest.mark.parametrize("threshold,angular", [(0.9995, False), (0.99995, True)])
def test_slerp_matches_the_formula_and_falls_back_to_a_line(family, threshold, angular):
    ckpt, patches, base = family
    pair = sorted(patches[:2], key=lambda p: p.id)
    values = [np.array([float(x) for x in exact(parent_values(p, base))]) for p in pair]
    dot = values[0] @ values[1] / np.linalg.norm(values[0]) / np.linalg.norm(values[1])
    assert (abs(dot) <= threshold) == angular
    t = 0.3
    theta = math.acos(dot)
    a, b = (math.sin((1 - t) * theta) / math.sin(theta), math.sin(t * theta) / math.sin(theta)) if angular else (1 - t, t)
    recipe = Recipe.make("slerp", [p.id for p in pair], None, t=t, threshold=threshold)
    got = result(ckpt, pair[::-1], recipe)
    v = [exact(x) for x in (parent_values(p, base) for p in pair)]
    cb = quantised(b)
    ca = quantised(a) if angular else UNIT - cb
    want = [nearest(TABLE, (ca * v[0][e] + cb * v[1][e]) / UNIT) for e in range(SIZE)]
    assert np.array_equal(got, np.array(want, dtype=np.uint16))


@pytest.mark.parametrize("method,weights,settings", [("task", [1, 0.5, 0.25], {}), ("ties", None, {"density": 0.4}),
                                                     ("dare", [0.5, 1, 1], {"seed": 5, "density": 0.6}),
                                                     ("model_stock", None, {})])
def test_every_method_writes_the_same_file_in_any_order(family, method, weights, settings, tmp_path):
    ckpt, patches, _ = family
    parents = patches[:3]
    ids = [p.id for p in parents]
    recipe = Recipe.make(method, ids, weights, **settings)
    seen = set()
    for order in itertools.permutations(range(3)):
        merge_with(ckpt, [parents[i] for i in order], recipe).patch.save(tmp_path / "m.neyra")
        seen.add((tmp_path / "m.neyra").read_bytes())
    assert len(seen) == 1


def test_a_merge_is_recomputed_from_its_recipe_and_tampering_shows(family, tmp_path):
    ckpt, patches, _ = family
    recipe = Recipe.make("ties", [p.id for p in patches], None, density=0.4)
    merged = merge_with(ckpt, patches, recipe).patch
    merged.save(tmp_path / "m.neyra")
    loaded = Patch.load(tmp_path / "m.neyra")
    assert verify(ckpt, loaded, patches[::-1]) == []

    loaded.meta["recipe"] = {**loaded.meta["recipe"], "params": {**loaded.meta["recipe"]["params"], "density": "0.5"}}
    assert verify(ckpt, loaded, patches) == ["w"]
    with pytest.raises(PatchMismatch):
        verify(ckpt, loaded, patches[:3])


def test_recipe_holds_no_float_and_names_weights_by_patch(family):
    _, patches, _ = family
    recipe = Recipe.make("ties", [p.id for p in patches], [1, 0.5, 1, 0.25], density=0.3)
    text = json.dumps(recipe.to_json())
    assert Recipe.from_json(json.loads(text)) == recipe
    assert all(not isinstance(v, float) for v in recipe.params.values())
    shuffled = Recipe.make("ties", [p.id for p in patches[::-1]], [0.25, 1, 0.5, 1], density=0.3)
    assert shuffled.id == recipe.id


def test_cli_merges_by_method_and_recomputes(family, tmp_path, model_dir, capsys):
    ckpt, patches, _ = family
    paths = []
    for i, p in enumerate(patches[:3]):
        p.save(tmp_path / f"p{i}.neyra")
        paths.append(str(tmp_path / f"p{i}.neyra"))
    base = str(ckpt.root) if hasattr(ckpt, "root") else str(tmp_path / "base")
    out = str(tmp_path / "ties.neyra")
    args = ["merge", base, *paths, "-o", out, "--method", "ties", "--weights", "1", "0.5", "1", "--set", "density=0.4"]
    assert main(args) in (0, 1)
    assert main(["recompute", base, out, *paths[::-1]]) == 0
    assert main(["recompute", base, out, *paths[:2]]) != 0
    assert main(["check", base, out, "--parents", *paths]) == 0
    assert main(["merge", base, *paths, "-o", out, "--method", "ties", "--set", "nope=1"]) == 2
    capsys.readouterr()


@pytest.mark.parametrize("kappa,cap", [(1.0, 1.0), (0.8, 1.0), (1.0, 0.3)])
def test_matched_takes_the_weight_that_keeps_the_displacement_of_the_parents(family, kappa, cap):
    ckpt, patches, base = family
    ordered = sorted(patches, key=lambda p: p.id)
    weights = [1.0, 0.5, 1.0, 0.5]
    b = np.array([float(x) for x in exact(base)])
    d = np.stack([np.array([float(x) for x in exact(parent_values(p, base))]) - b for p in ordered])
    w = np.array(weights)
    pull = min(cap, kappa * (w * np.linalg.norm(d, axis=1)).sum() / w.sum() / np.linalg.norm(w @ d))
    recipe = Recipe.make("matched", [p.id for p in ordered], weights, kappa=kappa, cap=cap)
    got = result(ckpt, ordered, recipe)
    assert np.array_equal(got, reference(base, ordered, lambda e, i: quantised(weights[i] * pull)))
    assert pull < 1 or cap == 1.0


def test_ties_siding_when_two_deltas_cancel_exactly(tmp_path, model_dir):
    # 1.0 moved by +2^-5 and -2^-5 sums to zero mass, and the tie goes to the positive side.
    base = bf16([1.0, 1.0, 1.0])
    ckpt = Checkpoint.open(str(model_dir("b", {"w": base})))
    up = make_patch(ckpt, Checkpoint.open(str(model_dir("u", {"w": bf16([1.0 + 2.0**-5, 1.0, 1.0])}))))
    down = make_patch(ckpt, Checkpoint.open(str(model_dir("d", {"w": bf16([1.0 - 2.0**-5, 1.0, 1.0])}))))
    out = result(ckpt, [up, down], Recipe.make("ties", [up.id, down.id], None, density=1.0))
    assert list(to_float(out, "BF16")) == [1.0 + 2.0**-5, 1.0, 1.0]


@pytest.mark.parametrize("method,settings", [("task", {}), ("dare", {"density": 0.6, "seed": 2}), ("matched", {}),
                                             ("model_stock", {}), ("ties", {"density": 0.25, "rescale": "l1"})])
def test_chunk_size_does_not_change_the_bytes(family, monkeypatch, method, settings):
    ckpt, patches, _ = family
    recipe = Recipe.make(method, [p.id for p in patches], None, **settings)
    whole = result(ckpt, patches, recipe)
    monkeypatch.setattr(methods, "CHUNK", 11)
    assert np.array_equal(result(ckpt, patches, recipe), whole)


def test_parents_read_from_checkpoints_merge_like_their_patches(family, tmp_path, model_dir):
    ckpt, patches, base = family
    models = [Checkpoint.open(str(model_dir(f"m{i}", {"w": parent_values(p, base), "v": ckpt.tensors["v"].read()})))
              for i, p in enumerate(patches)]
    parents = [methods.CheckpointParent(m, f"model{i}") for i, m in enumerate(models)]
    recipe = Recipe.make("ties", [p.id for p in parents], None, density=0.4)
    got = {name: bits for name, bits, _ in methods.stream(ckpt, parents[::-1], recipe)}
    again = Recipe.make("ties", [p.id for p in patches], None, density=0.4)
    assert np.array_equal(got["w"].reshape(-1), result(ckpt, patches, again).reshape(-1))
    assert "v" not in got
