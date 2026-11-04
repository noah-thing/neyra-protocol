"""Merge several patches made on one base, with the same bytes in any order.

Every float is an integer times a power of two, so the deltas can be added exactly as integers
and rounded to the tensor dtype once, to nearest even. Merging one rounded result into another
cannot do this: on the bf16 lattice (a + b) + c and a + (b + c) differ, so the merge is a function
of the set of patches, not of a sequence.

An entry only one patch touches already holds the exact sum, so it is copied. Arithmetic happens
where patches overlap. That is also where they can disagree, and the report says how often.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .numerics import from_float, is_float, to_float
from .patch import Patch, PatchMismatch, TensorEdit
from .protocol import bits_digest
from .store import Checkpoint

PRECISION = {"BF16": 8, "F16": 11, "F32": 24, "F64": 53}
LOWEST = {"BF16": -133, "F16": -24, "F32": -149, "F64": -1074}  # exponent of the smallest subnormal step
OVERFLOW = {"BF16": 128, "F16": 16, "F32": 128, "F64": 1024}  # first power of two the dtype cannot hold
SAFE = 52  # an int64 sum below 2^52 converts to float64 without losing a bit
ROUNDING = "rne/1"


@dataclass
class Overlap:
    name: str
    entries: int
    disagree: int


@dataclass
class Merged:
    patch: Patch
    overlaps: list[Overlap]


def split(bits: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw bits as (m, e) with value = m * 2^e and m an integer below 2^precision."""
    p = PRECISION[dtype]
    mant, exp = np.frexp(to_float(bits, dtype).astype(np.float64))
    return np.ldexp(mant, p).astype(np.int64), exp.astype(np.int64) - p


def round_exact(total: int, exp: int, dtype: str) -> float:
    """total * 2^exp rounded to the dtype, for sums too wide for int64. The result is exact in float64."""
    if total == 0:
        return 0.0
    mag = abs(total)
    drop = max(mag.bit_length() - PRECISION[dtype], LOWEST[dtype] - exp)
    if drop > 0:
        mag, rest = divmod(mag, 1 << drop)
        half = 1 << (drop - 1)
        mag += rest > half or (rest == half and mag & 1)
        exp += drop
    if mag.bit_length() + exp > OVERFLOW[dtype]:
        return math.copysign(math.inf, total)
    return math.copysign(math.ldexp(mag, exp), total)


def round_to_odd(x: np.ndarray) -> np.ndarray:
    """float64 to float32 keeping a sticky bit, so rounding on to bf16 is still a single rounding."""
    f = x.astype(np.float32)
    inexact = f.astype(np.float64) != x
    odd = (f.view(np.uint32) & 1).astype(bool)
    toward = np.where(x > f, np.float32(np.inf), np.float32(-np.inf))
    return np.where(inexact & ~odd, np.nextafter(f, toward), f)


def encode(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round float64 values that are exact sums to raw bits."""
    with np.errstate(over="ignore"):
        return from_float(round_to_odd(x) if dtype == "BF16" else x, dtype)


def exact_sum(dtype: str, values: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    """Raw bits of sum(weights * values) per entry, rounded once. Values are raw bits."""
    p = PRECISION[dtype]
    parts = [split(v, dtype) for v in values]
    alive = [(m != 0) & (w != 0) for (m, _), w in zip(parts, weights)]
    top = np.iinfo(np.int64).max
    low = np.minimum.reduce([np.where(a, e, top) for a, (_, e) in zip(alive, parts)])
    low = np.where(low == top, 0, low)
    spread = np.maximum.reduce([np.where(a, e - low, 0) for a, (_, e) in zip(alive, parts)])
    mass = sum(np.abs(w) for w in weights)
    fits = spread + p + np.ceil(np.log2(mass + 1)).astype(np.int64) <= SAFE

    total = np.zeros(len(low), dtype=np.int64)
    for (m, e), a, w in zip(parts, alive, weights):
        total += w * (m << np.where(fits & a, e - low, 0))
    with np.errstate(over="ignore"):
        x = np.ldexp(total.astype(np.float64), low)
    for i in np.flatnonzero(~fits):
        wide = sum(int(w[i]) * (int(m[i]) << int(e[i] - low[i])) for (m, e), a, w in zip(parts, alive, weights) if a[i])
        x[i] = round_exact(wide, int(low[i]), dtype)
    return encode(x, dtype)


def merge_tensor(name: str, dtype: str, base: np.ndarray, edits: list[TensorEdit]) -> tuple[np.ndarray, Overlap]:
    flat = base.reshape(-1)
    changes = []
    for e in edits:
        full = e.dense(base)
        if full.idx.size and (full.idx[0] < 0 or full.idx[-1] >= flat.size or np.any(np.diff(full.idx) <= 0)):
            raise PatchMismatch(f"{name}: edit indices must be strictly increasing and inside the tensor", "bad_index")
        moved = full.val != flat[full.idx]
        changes.append((full.idx[moved], full.val[moved]))
    out = flat.copy()
    for idx, val in changes:
        out[idx] = val
    entries, counts = np.unique(np.concatenate([idx for idx, _ in changes]), return_counts=True)
    shared = entries[counts >= 2]
    if not shared.size:
        return out.reshape(base.shape), Overlap(name, 0, 0)
    if not is_float(dtype):
        raise PatchMismatch(f"{name}: {dtype} entries edited by several patches cannot be summed", "not_mergeable")

    touched, values = [], []
    for idx, val in changes:
        at = np.minimum(np.searchsorted(idx, shared), max(len(idx) - 1, 0))
        hit = (idx[at] == shared) if len(idx) else np.zeros(len(shared), dtype=bool)
        touched.append(hit)
        values.append(np.where(hit, val[at] if len(idx) else flat[shared], flat[shared]))
    if not all(np.isfinite(to_float(v, dtype)).all() for v in [*values, flat[shared]]):
        raise PatchMismatch(f"{name}: overlapping entries hold NaN or infinity, so their sum is undefined",
                            "not_mergeable")
    touch, stacked = np.stack(touched), np.stack(values)
    first = stacked[touch.argmax(axis=0), np.arange(len(shared))]
    disagree = int((touch & (stacked != first)).any(axis=0).sum())

    count = touch.sum(axis=0).astype(np.int64)
    out[shared] = exact_sum(dtype, [*values, flat[shared]], [*touch.astype(np.int64), 1 - count])
    return out.reshape(base.shape), Overlap(name, len(shared), disagree)


def merge(base: Checkpoint, patches: list[Patch]) -> Merged:
    ids = [p.id for p in patches]
    if len(set(ids)) != len(ids):
        raise PatchMismatch("the same patch was given twice, which would count its edit twice", "duplicate_patch")
    table = patches[0].check_base(base)
    if any(p.base_root != patches[0].base_root for p in patches):
        raise PatchMismatch("these patches were not made on the same base", "base_mismatch")

    by_name: dict[str, list[TensorEdit]] = {}
    for p in patches:
        for e in p.edits:
            by_name.setdefault(e.name, []).append(e)
    edits, overlaps = [], []
    for name in sorted(by_name):
        ref = base.tensors[name]
        bits = ref.read()
        out, overlap = merge_tensor(name, ref.dtype, bits, by_name[name])
        overlaps.append(overlap)
        idx = np.flatnonzero(out.reshape(-1) != bits.reshape(-1))
        if idx.size:
            edits.append(TensorEdit(name, ref.dtype, ref.shape, by_name[name][0].base_digest,
                                    bits_digest(ref.dtype, out), idx, out.reshape(-1)[idx]))
    merged = Patch(edits, {"merged_from": sorted(ids), "rounding": ROUNDING}).bind(table)
    return Merged(merged, [o for o in overlaps if o.entries])
