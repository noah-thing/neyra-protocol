import itertools
from fractions import Fraction

import numpy as np
import pytest
from conftest import bf16
from hypothesis import given, settings
from hypothesis import strategies as st

from neyra.cli import main
from neyra.merge import encode, exact_sum, merge, merge_tensor, round_exact
from neyra.numerics import from_float, to_float
from neyra.patch import Patch, PatchMismatch, TensorEdit, make_patch
from neyra.store import Checkpoint


def edit(base, moved):
    """A sparse edit of a flat bf16 tensor, as {index: new bits}."""
    idx = np.array(sorted(moved), dtype=np.int64)
    return TensorEdit("w", "BF16", base.shape, "", "", idx, np.array([moved[i] for i in idx], dtype=np.uint16))


def sequential(base, deltas):
    """What merging one result into the next does: round after every patch."""
    out = to_float(base, "BF16").astype(np.float64)
    for d in deltas:
        out = to_float(from_float((out + d).astype(np.float32), "BF16"), "BF16").astype(np.float64)
    return from_float(out.astype(np.float32), "BF16")


def lattice(dtype):
    """Every finite value of a 16 bit float as an exact fraction, with infinity one step past the largest."""
    bits = np.arange(65536, dtype=np.uint16)
    with np.errstate(invalid="ignore"):
        values = to_float(bits, dtype)
    finite = [(Fraction(float(v)), int(b)) for b, v in zip(bits, values) if np.isfinite(v) and not b & 0x8000]
    finite.sort()
    top, last = finite[-1][0], finite[-1][0] - finite[-2][0]
    inf = {"BF16": 0x7F80, "F16": 0x7C00}[dtype]
    return [*finite, (top + last, inf)]


def nearest(table, x):
    """Round to nearest even by brute force, as an independent check on the integer arithmetic."""
    sign = 0x8000 if x < 0 else 0
    mag = abs(x)
    if mag == 0:
        return 0
    values = [v for v, _ in table]
    i = max(min(np.searchsorted(np.array(values, dtype=object), mag), len(table) - 1), 1)
    (lo, lo_bits), (hi, hi_bits) = table[i - 1], table[i]
    if mag - lo == hi - mag:
        return sign | (lo_bits if lo_bits % 2 == 0 else hi_bits)
    return sign | (lo_bits if mag - lo < hi - mag else hi_bits)


def test_three_way_counterexample_from_the_lattice_note():
    # 1 - 2^-8 is on the bf16 grid, and each branch lands on 1.0, which loses half the step.
    base = bf16([1 - 2.0**-8, 0.5, -0.5, 2.0, 3.0, 4.0])
    step = 2.0**-8
    edits = [edit(base, {0: bf16([1.0])[0], 2 + i: bf16([float(i + 5)])[0]}) for i in range(3)]
    out, overlap = merge_tensor("w", "BF16", base, edits)
    assert to_float(out, "BF16")[0] == 1 + 2.0**-7
    assert (overlap.entries, overlap.disagree) == (1, 0)

    deltas = [np.array([step, 0, 0, 0, 0, 0])] * 3
    nested = sequential(base, deltas)
    assert to_float(nested, "BF16")[0] == 1.0
    one_pair = sequential(base, deltas[:2])
    assert to_float(one_pair, "BF16")[0] == 1.0


def test_order_never_changes_the_bytes_but_sequential_merging_does(tmp_path, model_dir, rng):
    w = bf16(rng.standard_normal((64, 64)) * 0.02)
    f = to_float(w, "BF16").astype(np.float64)
    base = Checkpoint.open(str(model_dir("a", {"w": w})))
    patches, deltas = [], []
    for i in range(4):
        edited = bf16(f + rng.normal(0, 1, f.shape) * np.abs(f) * 2.0**-8 * 3 * (rng.random(f.shape) < 0.5))
        deltas.append(to_float(edited, "BF16").astype(np.float64) - f)
        patches.append(make_patch(base, Checkpoint.open(str(model_dir(f"e{i}", {"w": edited})))))

    files = set()
    for order in itertools.permutations(range(4)):
        merge(base, [patches[i] for i in order]).patch.save(tmp_path / "m.neyra")
        files.add((tmp_path / "m.neyra").read_bytes())
    assert len(files) == 1

    merged = Patch.load(tmp_path / "m.neyra").apply(base)["w"]
    differing = [np.mean(sequential(w, [deltas[i] for i in o]) != merged) for o in itertools.permutations(range(4))]
    assert max(differing) > 0


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 2**32 - 1), st.permutations(range(4)))
def test_any_order_gives_identical_entries(seed, order):
    rng = np.random.default_rng(seed)
    base = bf16(rng.standard_normal(200) * 0.05)
    f = to_float(base, "BF16").astype(np.float64)
    edits = []
    for _ in range(4):
        pick = np.flatnonzero(rng.random(200) < 0.4)
        moved = bf16(f[pick] + rng.normal(0, 2, len(pick)) * np.abs(f[pick]) * 2.0**-8)
        edits.append(edit(base, dict(zip(pick.tolist(), moved))))
    forward, _ = merge_tensor("w", "BF16", base, edits)
    shuffled, _ = merge_tensor("w", "BF16", base, [edits[i] for i in order])
    assert forward.tobytes() == shuffled.tobytes()


LAYOUT = {"BF16": (7, 8), "F16": (10, 5)}  # mantissa bits, exponent bits


def near_exponent(rng, dtype, center, window, n):
    """Random finite-looking values whose exponent field sits in a narrow window."""
    mantissa, exponent = LAYOUT[dtype]
    field = np.clip(center + rng.integers(0, window, n), 0, 2**exponent - 2)
    return ((rng.integers(0, 2, n) << 15) | (field << mantissa) | rng.integers(0, 2**mantissa, n)).astype(np.uint16)


@pytest.mark.parametrize("dtype", ["BF16", "F16"])
def test_sum_matches_brute_force_rounding(dtype, rng):
    table = lattice(dtype)
    top = 2 ** LAYOUT[dtype][1] - 2
    for center, window in itertools.product([0, 1, top // 2, top - 3, top], [3, 12, 40, 120]):
        n = 120
        bits = [near_exponent(rng, dtype, center, window, n) for _ in range(3)]
        weights = [rng.integers(-2, 3, n).astype(np.int64) for _ in range(3)]
        got = exact_sum(dtype, bits, weights)
        for i in range(n):
            exact = sum(int(w[i]) * Fraction(float(to_float(b[i : i + 1], dtype)[0])) for b, w in zip(bits, weights))
            assert got[i] == nearest(table, exact), (dtype, center, window, i)


@pytest.mark.parametrize("dtype", ["BF16", "F16", "F32"])
def test_wide_and_narrow_paths_round_alike(dtype, rng):
    for _ in range(2000):
        total, exp = int(rng.integers(-2**51, 2**51)), int(rng.integers(-170, 120))
        narrow = encode(np.array([np.ldexp(float(total), exp)]), dtype)
        wide = encode(np.array([round_exact(total, exp, dtype)]), dtype)
        assert narrow[0] == wide[0], (dtype, total, exp)


def test_spread_beyond_int64_is_still_exact():
    # 2^100 + 2^-100 - 2^100 needs 200 bits, and only the small term survives.
    big, tiny = bf16([2.0**100]), bf16([2.0**-100])
    got = exact_sum("BF16", [big, tiny, big], [np.array([1]), np.array([1]), np.array([-1])])
    assert to_float(got, "BF16")[0] == 2.0**-100


def test_overlap_report_separates_agreement_from_conflict(rng):
    base = bf16(np.ones(8))
    one = edit(base, {0: bf16([2.0])[0], 1: bf16([3.0])[0], 5: bf16([4.0])[0]})
    two = edit(base, {0: bf16([2.0])[0], 1: bf16([0.5])[0], 6: bf16([4.0])[0]})
    out, overlap = merge_tensor("w", "BF16", base, [one, two])
    assert (overlap.entries, overlap.disagree) == (2, 1)
    # Equal edits are added twice, not coalesced: 1 + 1 + 1.
    assert list(to_float(out, "BF16")[:2]) == [3.0, 2.5]
    assert to_float(out, "BF16")[5] == 4.0 and to_float(out, "BF16")[6] == 4.0


def test_patches_that_cancel_leave_the_tensor_alone(tmp_path, model_dir, rng):
    w = bf16(np.arange(4, 36).reshape(4, 8))
    up, down = w.copy(), w.copy()
    up[0, 0] = bf16([float(to_float(w, "BF16")[0, 0]) + 1])[0]
    down[0, 0] = bf16([float(to_float(w, "BF16")[0, 0]) - 1])[0]
    base = Checkpoint.open(str(model_dir("a", {"w": w})))
    patches = [make_patch(base, Checkpoint.open(str(model_dir(n, {"w": t})))) for n, t in (("u", up), ("d", down))]
    merged = merge(base, patches)
    assert merged.patch.edits == [] and merged.patch.result_root == merged.patch.base_root


def test_refuses_repeats_and_foreign_bases(tmp_path, model_dir, rng):
    w = bf16(rng.standard_normal((4, 8)))
    other = w.copy()
    other[0, 0] = bf16([9.0])[0]
    base = Checkpoint.open(str(model_dir("a", {"w": w})))
    p = make_patch(base, Checkpoint.open(str(model_dir("b", {"w": other}))))
    with pytest.raises(PatchMismatch):
        merge(base, [p, p])
    stranger = Checkpoint.open(str(model_dir("c", {"w": bf16(rng.standard_normal((4, 8)))})))
    with pytest.raises(PatchMismatch):
        merge(stranger, [p, p.invert(base)])


def test_nan_in_an_overlap_is_refused():
    base = bf16([1.0, 2.0])
    nan = np.array([0x7FC0], dtype=np.uint16)
    with pytest.raises(PatchMismatch):
        merge_tensor("w", "BF16", base, [edit(base, {0: nan[0]}), edit(base, {0: bf16([3.0])[0]})])


def test_cli_merge_is_order_independent_and_reports_overlap(tmp_path, model_dir, rng, capsys):
    w = bf16(rng.standard_normal((16, 16)) * 0.1)
    base = model_dir("a", {"w": w})
    paths = []
    for i in range(3):
        e = w.copy()
        e[i : i + 8, :4] = bf16(to_float(w, "BF16")[i : i + 8, :4] + 0.01 * (i + 1))
        model_dir(f"e{i}", {"w": e})
        main(["patch", str(base), str(tmp_path / f"e{i}"), "-o", str(tmp_path / f"p{i}.neyra")])
        paths.append(str(tmp_path / f"p{i}.neyra"))
    capsys.readouterr()
    assert main(["merge", str(base), *paths, "-o", str(tmp_path / "a.neyra")]) == 1
    assert "entries touched by more than one patch" in capsys.readouterr().out
    assert main(["merge", str(base), *paths[::-1], "-o", str(tmp_path / "b.neyra")]) == 1
    assert (tmp_path / "a.neyra").read_bytes() == (tmp_path / "b.neyra").read_bytes()
    assert main(["apply", str(base), str(tmp_path / "a.neyra"), "-o", str(tmp_path / "out")]) == 0


def test_cli_merge_of_disjoint_patches_is_clean(tmp_path, model_dir, rng):
    w = bf16(rng.standard_normal((8, 8)))
    base = model_dir("a", {"w": w})
    paths = []
    for i in range(2):
        e = w.copy()
        e[i] = bf16(to_float(w, "BF16")[i] + 1)
        model_dir(f"e{i}", {"w": e})
        main(["patch", str(base), str(tmp_path / f"e{i}"), "-o", str(tmp_path / f"p{i}.neyra")])
        paths.append(str(tmp_path / f"p{i}.neyra"))
    assert main(["merge", str(base), *paths, "-o", str(tmp_path / "m.neyra")]) == 0
    assert main(["merge", str(base), paths[0], "-o", str(tmp_path / "m.neyra")]) == 2


def test_indices_out_of_order_are_refused():
    base = bf16(np.ones(8))
    bad = TensorEdit("w", "BF16", base.shape, "", "", np.array([5, 2], dtype=np.int64), bf16(np.array([2.0, 3.0])))
    with pytest.raises(PatchMismatch) as caught:
        merge_tensor("w", "BF16", base, [bad])
    assert caught.value.code == "bad_index"
