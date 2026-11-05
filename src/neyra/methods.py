"""Merge methods that keep the exactness of `merge`: task arithmetic, TIES, DARE, SLERP and Model Stock.

Each method decides, for every entry, how much of each parent to take. The decision is a set of integer
coefficients on a grid of 2^-16, and the result is the exact sum of coefficient times parent value, rounded once
to the tensor dtype. Where a method divides (TIES by the number of agreeing parents, Model Stock by 1 + (n-1)cos)
the quotient is rounded to the grid before the sum. That rounding is part of the recipe, it is smaller than
2^-17 of a delta, and it is the only inexact step.

Selection (which entries survive a trim, which sign wins) looks at the parents in order of patch id, never in
the order they were given, and the recipe names weights by patch id. DARE's drop mask is a hash of the seed,
the tensor name, the patch id and the flat index, so it does not depend on a generator state or a library.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from .merge import Merged, Overlap, exact_sum
from .numerics import is_float, to_float
from .patch import Patch, PatchMismatch, TensorEdit
from .protocol import bits_digest, digest, jcs
from .store import Checkpoint

GRID = 16
UNIT = 1 << GRID
CHUNK = 1 << 20
VERSION = "neyra.recipe/1"
DEFAULTS = {
    "task": {"lam": 1.0, "normalize": False},
    "matched": {"kappa": 1.0, "cap": 1.0},
    "ties": {"density": 0.5, "lam": 1.0, "normalize": True, "consensus": "sum", "rescale": "none"},
    "dare": {"density": 0.5, "lam": 1.0, "normalize": False, "consensus": "none", "rescale": "density", "seed": 0},
    "slerp": {"t": 0.5, "threshold": 0.9995, "first": None},
    "model_stock": {},
}


def decimal(x: float) -> str:
    """Parameters are stored as strings so a recipe holds no float and can be hashed."""
    return repr(float(x))


@dataclass(frozen=True)
class Recipe:
    method: str
    parents: tuple[str, ...]
    weights: dict[str, str]
    params: dict

    @classmethod
    def make(cls, method: str, ids: list[str], weights: list[float] | None = None, **params) -> Recipe:
        if method not in DEFAULTS:
            raise PatchMismatch(f"unknown merge method {method!r}, choose from {sorted(DEFAULTS)}", "bad_recipe")
        unknown = set(params) - set(DEFAULTS[method])
        if unknown:
            raise PatchMismatch(f"{method} has no parameter {sorted(unknown)}", "bad_recipe")
        weights = weights or [1.0] * len(ids)
        if len(weights) != len(ids):
            raise PatchMismatch("there must be one weight per patch", "bad_recipe")
        merged = {**DEFAULTS[method], **params}
        for key, value in merged.items():
            if isinstance(value, float):
                merged[key] = decimal(value)
        if method == "slerp":
            if len(ids) != 2:
                raise PatchMismatch("slerp needs exactly two patches", "bad_recipe")
            # the order of interpolation is part of the recipe; the patch order given on the command line is not
            merged["first"] = merged["first"] or min(ids)
        return cls(method, tuple(sorted(ids)), {i: decimal(w) for i, w in zip(ids, weights)}, merged)

    def param(self, key: str):
        value = self.params[key]
        return float(value) if isinstance(value, str) and key not in ("consensus", "rescale", "first") else value

    def weight(self, pid: str) -> float:
        return float(self.weights[pid])

    def to_json(self) -> dict:
        return {"version": VERSION, "method": self.method, "parents": list(self.parents),
                "weights": dict(sorted(self.weights.items())), "params": dict(sorted(self.params.items()))}

    @classmethod
    def from_json(cls, doc: dict) -> Recipe:
        if doc.get("version") != VERSION:
            raise PatchMismatch(f"recipe version {doc.get('version')!r} is not supported", "bad_recipe")
        return cls(doc["method"], tuple(doc["parents"]), dict(doc["weights"]), dict(doc["params"]))

    @property
    def id(self) -> str:
        return digest("recipe", jcs(self.to_json()))


def key64(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("\0".join(parts).encode()).digest()[:8], "little")


def mix64(x: np.ndarray) -> np.ndarray:
    """The splitmix64 finaliser. uint64 arithmetic wraps, which is the point."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def keep_hash(seed: int, tensor: str, patch: str, idx: np.ndarray, density: float) -> np.ndarray:
    """Whether each flat index survives DARE's drop. Independent per (seed, tensor, patch, index)."""
    stream = np.uint64(mix64(np.array([key64(str(seed), tensor, patch)], dtype=np.uint64))[0])
    with np.errstate(over="ignore"):
        u = mix64(idx.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + stream)
    return u < np.uint64(min(int(density * 2**64), 2**64 - 1))


def quantise(c: np.ndarray) -> np.ndarray:
    return np.rint(c * UNIT).astype(np.int64)


@dataclass
class Column:
    """One parent's whole tensor as raw bits, flat."""
    bits: np.ndarray

    def chunk(self, lo: int, hi: int) -> np.ndarray:
        return self.bits[lo:hi]


class Parent:
    """Something a merge takes values from. `id` names it in the recipe."""
    id: str

    def tensor(self, name: str, base: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError


class PatchParent(Parent):
    def __init__(self, patch: Patch):
        self.patch, self.id = patch, patch.id
        self.edits = {e.name: e for e in patch.edits}

    def tensor(self, name: str, base: np.ndarray) -> np.ndarray | None:
        edit = self.edits.get(name)
        return None if edit is None else edit.apply(base.reshape(edit.shape), verify=False).reshape(-1)


class CheckpointParent(Parent):
    """A fine tuned model read straight from its files, for models whose patch would be larger than they are."""

    def __init__(self, ckpt: Checkpoint, ident: str):
        self.ckpt, self.id = ckpt, ident

    def tensor(self, name: str, base: np.ndarray) -> np.ndarray | None:
        ref = self.ckpt.tensors.get(name)
        if ref is None:
            return None
        bits = ref.read().reshape(-1)
        return None if np.array_equal(bits, base) else bits


@dataclass
class Plan:
    """Per tensor numbers a method needs before it can look at a chunk."""
    recipe: Recipe
    name: str
    dtype: str
    size: int
    base: np.ndarray
    cols: list[Column]
    ids: list[str]
    thr: list[np.float32] = field(default_factory=list)
    quota: list[int] = field(default_factory=list)
    ties_before: list[np.ndarray] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    stock: float = 0.0
    pull: float = 1.0
    angle: tuple[float, float, bool] = (0.0, 0.0, True)

    def gather(self, lo: int, hi: int, dense: bool = False):
        """Flat indices to work on in [lo, hi), their deltas (parents x entries) and the raw bits of every parent."""
        bits = [c.chunk(lo, hi) for c in self.cols]
        base_f = to_float(self.base[lo:hi], self.dtype).astype(np.float64)
        delta = np.stack([to_float(b, self.dtype).astype(np.float64) - base_f for b in bits])
        sel = np.arange(hi - lo) if dense else np.flatnonzero((delta != 0).any(axis=0))
        return lo + sel, delta[:, sel], [b[sel] for b in bits]


def prepare(plan: Plan) -> None:
    r = plan.recipe
    n = len(plan.cols)
    plan.scale = [1.0] * n
    if r.method == "ties":
        trim_thresholds(plan)
    if r.method in ("ties", "dare"):
        mode = r.params["rescale"]
        if mode == "density":
            plan.scale = [1.0 / r.param("density")] * n
        elif mode == "l1":
            before, after = [[] for _ in range(n)], [[] for _ in range(n)]
            for lo in range(0, plan.size, CHUNK):
                idx, d, _ = plan.gather(lo, min(lo + CHUNK, plan.size))
                for i in range(n):
                    keep = survivors(plan, i, d[i], idx)
                    before[i].append(blocked(np.abs(d[i]))), after[i].append(blocked(np.abs(d[i][keep])))
            plan.scale = [math.fsum(b) / math.fsum(a) if math.fsum(a) > 0 else 1.0 for a, b in zip(after, before)]
    elif r.method == "matched":
        plan.pull = matched_pull(plan)
    elif r.method == "model_stock":
        plan.stock = stock_t(plan)
    elif r.method == "slerp":
        plan.angle = slerp_coefficients(plan)


def trim_thresholds(plan: Plan) -> None:
    """Per parent: the magnitude of the k-th largest delta and how many entries tied at it are kept."""
    density = plan.recipe.param("density")
    k = max(1, int(density * plan.size)) if density > 0 else 0
    chunks = plan.size // CHUNK + 1
    def mags(col: Column, lo: int) -> np.ndarray:
        hi = min(lo + CHUNK, plan.size)
        d = to_float(col.chunk(lo, hi), plan.dtype).astype(np.float64) - to_float(plan.base[lo:hi], plan.dtype)
        return np.abs(d[d != 0]).astype(np.float32)

    for col in plan.cols:
        found = np.concatenate([mags(col, lo) for lo in range(0, plan.size, CHUNK)])
        if k >= len(found):
            thr, quota = np.float32(0), 0
        elif k == 0:
            thr, quota = np.float32(np.inf), 0
        else:
            found.partition(len(found) - k)
            thr = found[len(found) - k]
            quota = k - int((found > thr).sum())
        del found
        counts = np.array([int((mags(col, lo) == thr).sum()) if thr > 0 else 0 for lo in range(0, plan.size, CHUNK)])
        plan.thr.append(thr)
        plan.quota.append(quota)
        plan.ties_before.append(np.concatenate([[0], np.cumsum(counts), np.zeros(chunks - len(counts))])[:chunks].astype(np.int64))


def survivors(plan: Plan, i: int, delta: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Which entries of parent i the method keeps. Entries are in index order and idx lies in one chunk."""
    method = plan.recipe.method
    if method == "dare":
        return keep_hash(int(plan.recipe.param("seed")), plan.name, plan.ids[i], idx, plan.recipe.param("density"))
    if method == "ties":
        mag = np.abs(delta).astype(np.float32)
        tie = (mag == plan.thr[i]) & (mag > 0)
        # ties are kept in index order, so the cut does not depend on how a sort breaks them
        rank = np.cumsum(tie) + (plan.ties_before[i][idx[0] // CHUNK] if len(idx) else 0)
        return (mag > plan.thr[i]) | (tie & (rank <= plan.quota[i]))
    return np.ones(delta.shape, dtype=bool)


def blocked(x: np.ndarray) -> float:
    """Sum of a float64 array that does not depend on how numpy blocks the reduction."""
    pad = -len(x) % 4096
    return math.fsum(np.concatenate([x, np.zeros(pad)]).reshape(-1, 4096).sum(axis=1))


def matched_pull(plan: Plan) -> float:
    """The factor that makes the merged displacement as long as the parents' own, as measured in this tensor.

    Parents that agree add up to more than any of them, parents that do not agree add up to less, and the
    overshoot is what makes plain task arithmetic need a hand tuned weight. The overlap of the parents is what
    `merge` already reports, here it sets the weight of every parent in the tensor.
    """
    r = plan.recipe
    w = np.array([r.weight(i) for i in plan.ids])
    own = [[] for _ in plan.cols]
    total = []
    for lo in range(0, plan.size, CHUNK):
        _, d, _ = plan.gather(lo, min(lo + CHUNK, plan.size))
        for i in range(len(own)):
            own[i].append(blocked(d[i] * d[i]))
        total.append(blocked((w[:, None] * d).sum(axis=0) ** 2))
    target = sum(abs(wi) * math.sqrt(math.fsum(o)) for wi, o in zip(w, own)) / max(np.abs(w).sum(), 1e-8)
    reach = math.sqrt(math.fsum(total))
    if reach == 0:
        return 0.0
    return min(r.param("cap"), r.param("kappa") * target / reach)


def stock_t(plan: Plan) -> float:
    n = len(plan.cols)
    if n < 2:
        raise PatchMismatch("model_stock needs at least two patches", "bad_recipe")
    parts: dict[tuple[int, int], list[float]] = {(i, j): [] for i in range(n) for j in range(i, n)}
    for lo in range(0, plan.size, CHUNK):
        _, d, _ = plan.gather(lo, min(lo + CHUNK, plan.size))
        for (i, j), acc in parts.items():
            acc.append(blocked(d[i] * d[j]))
    dot = {k: math.fsum(v) for k, v in parts.items()}
    cos = [float(np.clip(dot[i, j] / max(math.sqrt(dot[i, i] * dot[j, j]), 1e-6), -1, 1))
           for i in range(n) for j in range(i + 1, n)]
    c = math.fsum(cos) / len(cos)
    denom = 1 + (n - 1) * c
    return 0.0 if abs(denom) < 1e-6 else n * c / denom


def slerp_coefficients(plan: Plan) -> tuple[float, float, bool]:
    """Weights on the first and second parent, and whether the interpolation fell back to linear."""
    r = plan.recipe
    a_col, b_col = (plan.cols[plan.ids.index(r.params["first"])], plan.cols[1 - plan.ids.index(r.params["first"])])
    t = r.param("t")
    aa, bb, ab = [], [], []
    for lo in range(0, plan.size, CHUNK):
        hi = min(lo + CHUNK, plan.size)
        a = to_float(a_col.chunk(lo, hi), plan.dtype).astype(np.float64)
        b = to_float(b_col.chunk(lo, hi), plan.dtype).astype(np.float64)
        aa.append(blocked(a * a)), bb.append(blocked(b * b)), ab.append(blocked(a * b))
    na, nb = math.sqrt(math.fsum(aa)), math.sqrt(math.fsum(bb))
    na, nb = (na if na > 1e-8 else 1.0), (nb if nb > 1e-8 else 1.0)
    dot = float(np.clip(math.fsum(ab) / (na * nb), -1, 1))
    if abs(dot) > r.param("threshold"):
        return 1 - t, t, True
    theta = math.acos(dot)
    return math.sin((1 - t) * theta) / math.sin(theta), math.sin(t * theta) / math.sin(theta), False


def coefficients(plan: Plan, d: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer coefficients, in units of 2^-16, on each parent's value and on the base value, per entry."""
    r = plan.recipe
    n = len(plan.cols)
    if r.method == "model_stock":
        coef = np.full(d.shape, quantise(np.array(plan.stock / n)))
        return coef, UNIT - coef.sum(axis=0)
    if r.method == "slerp":
        first = plan.ids.index(r.params["first"])
        a, b, linear = plan.angle
        coef = np.zeros(d.shape, dtype=np.int64)
        coef[1 - first] = quantise(np.array(b))
        coef[first] = UNIT - coef[1 - first] if linear else quantise(np.array(a))
        return coef, np.zeros(d.shape[1], dtype=np.int64)

    w = np.array([r.weight(i) for i in plan.ids])[:, None]
    if r.method == "matched":
        coef = quantise(np.repeat(w * plan.pull, d.shape[1], axis=1))
        return coef, UNIT - coef.sum(axis=0)
    keep = np.stack([survivors(plan, i, d[i], idx) for i in range(n)])
    scale = np.array(plan.scale)[:, None]
    consensus = r.params["consensus"] if r.method != "task" else "none"
    if consensus == "none":
        divisor = np.full(d.shape[1], w.sum())
        divisor[np.abs(divisor) < 1e-8] = 1.0
        agree = np.ones(d.shape, dtype=bool)
    else:
        shaped = np.where(keep, d * scale * w, 0.0)
        mass = shaped.sum(axis=0) if consensus == "sum" else np.sign(shaped).sum(axis=0)
        agree = np.sign(shaped) == np.where(mass >= 0, 1.0, -1.0)
        divisor = (w * agree).sum(axis=0)
        divisor[divisor == 0] = 1.0
    c = np.where(agree & keep, w * scale, 0.0) * r.param("lam")
    if r.params["normalize"]:
        c = c / divisor
    coef = quantise(c)
    return coef, UNIT - coef.sum(axis=0)


def merge_tensor(plan: Plan) -> tuple[np.ndarray, Overlap]:
    out = plan.base.copy()
    prepare(plan)
    dense = plan.recipe.method == "slerp" and not plan.angle[2]
    shared = differ = 0
    for lo in range(0, plan.size, CHUNK):
        idx, d, bits = plan.gather(lo, min(lo + CHUNK, plan.size), dense)
        if not idx.size:
            continue
        several = (d != 0).sum(axis=0) >= 2
        shared += int(several.sum())
        differ += int((several & (np.stack(bits) != bits[0]).any(axis=0)).sum())
        coef, rest = coefficients(plan, d, idx)
        out[idx] = exact_sum(plan.dtype, [*bits, plan.base[idx]], [*coef, rest], -GRID)
    return out, Overlap(plan.name, shared, differ)


def stream(base: Checkpoint, parents: list[Parent], recipe: Recipe):
    """Yield (name, merged bits, overlap) for every tensor some parent changed. Parents may be given in any order."""
    if len({p.id for p in parents}) != len(parents):
        raise PatchMismatch("the same parent was given twice, which would count its edit twice", "duplicate_patch")
    if tuple(sorted(p.id for p in parents)) != recipe.parents:
        raise PatchMismatch("the parents are not the ones named in the recipe", "bad_recipe")
    ordered = sorted(parents, key=lambda p: p.id)
    for name in sorted(base.tensors):
        ref = base.tensors[name]
        bits = ref.read().reshape(-1)
        found = [p.tensor(name, bits) for p in ordered]
        if all(f is None for f in found):
            continue
        if not is_float(ref.dtype) or ref.dtype == "F64":
            if sum(f is not None for f in found) > 1:
                raise PatchMismatch(f"{name}: {ref.dtype} entries edited by several patches cannot be combined", "not_mergeable")
            yield name, next(f for f in found if f is not None), Overlap(name, 0, 0)
            continue
        cols = [Column(bits if f is None else f) for f in found]
        if not all(np.isfinite(to_float(c.bits, ref.dtype)).all() for c in cols):
            raise PatchMismatch(f"{name}: edited entries hold NaN or infinity, so their combination is undefined", "not_mergeable")
        out, overlap = merge_tensor(Plan(recipe, name, ref.dtype, bits.size, bits, cols, [p.id for p in ordered]))
        yield name, out, overlap


def merge_with(base: Checkpoint, patches: list[Patch], recipe: Recipe) -> Merged:
    """Merge patches made on one base by a recipe. The same set of patches gives the same bytes in any order."""
    table = patches[0].check_base(base)
    if any(p.base_root != patches[0].base_root for p in patches):
        raise PatchMismatch("these patches were not made on the same base", "base_mismatch")
    edits, overlaps = [], []
    for name, result, overlap in stream(base, [PatchParent(p) for p in patches], recipe):
        ref = base.tensors[name]
        bits = ref.read().reshape(-1)
        overlaps.append(overlap)
        idx = np.flatnonzero(result != bits)
        if idx.size:
            edits.append(TensorEdit(name, ref.dtype, ref.shape, table[name]["digest"],
                                    bits_digest(ref.dtype, result.reshape(ref.shape)), idx, result[idx]))
    meta = {"merged_from": sorted(p.id for p in patches), "rounding": "rne/1", "recipe": recipe.to_json(),
            "recipe_id": recipe.id}
    return Merged(Patch(edits, meta).bind(table), [o for o in overlaps if o.entries])


def verify(base: Checkpoint, merged: Patch, parents: list[Patch]) -> list[str]:
    """Recompute a merge from the recipe it carries. Returns the tensors that differ, empty when it is bit for bit."""
    doc = merged.meta.get("recipe")
    if doc is None:
        raise PatchMismatch("this patch carries no recipe", "no_recipe")
    again = merge_with(base, parents, Recipe.from_json(doc)).patch
    mine = {e.name: e.result_digest for e in merged.edits}
    theirs = {e.name: e.result_digest for e in again.edits}
    return sorted(k for k in mine.keys() | theirs.keys() if mine.get(k) != theirs.get(k))
