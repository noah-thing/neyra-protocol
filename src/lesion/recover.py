"""Recover what an edit did to a tensor from the tensor alone.

Given W and W', find the support of the change and its form: rounding noise,
zeroing, a scale, a low rank update, or a projection W' = W - a u(u^T W) with its
direction and strength. Published edits rebuild to within bf16 rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .numerics import is_float, to_float

MANTISSA = {"BF16": 8, "F16": 11, "F32": 24, "F64": 53}
MAX_RANK = 32
EXACT_SVD_LIMIT = 2048


@dataclass
class Structure:
    name: str
    shape: tuple[int, ...]
    dtype: str
    changed: int
    rows: int = 0
    cols: int = 0
    relative: float = 0.0
    form: str = "dense"
    rank: int | None = None
    energy: float | None = None
    alpha: float | None = None
    side: str | None = None
    residual: float | None = None
    direction: np.ndarray | None = field(default=None, repr=False)
    support: np.ndarray | None = field(default=None, repr=False)

    @property
    def recipe(self) -> str:
        where = f"{self.cols} of {self.shape[-1]} columns" if self.cols < self.shape[-1] else "all columns"
        if len(self.shape) == 2 and self.rows < self.shape[0]:
            where = f"{self.rows} of {self.shape[0]} rows, {where}"
        if self.form == "projection":
            term = "u(uᵀW)" if self.side == "output" else "(Wv)vᵀ"
            return f"W' = W - {self.alpha:.3f} {term} on {where}"
        if self.form == "rank":
            return f"rank {self.rank} update on {where}"
        if self.form == "zeroed":
            return f"zeroed {where}"
        if self.form == "scaled":
            return f"W' = {self.alpha:.4f} W"
        if self.form == "rounding":
            return "rounding only, every change is one unit in the last place"
        return f"{self.form}, {self.changed:,} entries"


def _top_singular(d: np.ndarray, k: int = MAX_RANK):
    """Exact below the limit, randomized above it. Returns U, s, Vh for the top k."""
    if min(d.shape) <= EXACT_SVD_LIMIT:
        u, s, vh = np.linalg.svd(d, full_matrices=False)
        return u[:, :k], s, vh[:k]
    rng = np.random.default_rng(0)
    y = d @ rng.standard_normal((d.shape[1], k + 16))
    for _ in range(3):
        y = d @ (d.T @ np.linalg.qr(y)[0])
    q = np.linalg.qr(y)[0]
    u, s, vh = np.linalg.svd(q.T @ d, full_matrices=False)
    return (q @ u)[:, :k], s, vh[:k]


def _rounding_sigma(edited: np.ndarray, dtype: str) -> float:
    """Per entry standard deviation of the error from storing W' in its dtype."""
    return float(np.sqrt(np.mean(edited**2))) * 2.0 ** (1 - MANTISSA[dtype]) / np.sqrt(12)


def _one_ulp(base_bits: np.ndarray, edited_bits: np.ndarray, mask: np.ndarray) -> bool:
    gap = np.abs(base_bits[mask].astype(np.int64) - edited_bits[mask].astype(np.int64))
    return gap.size > 0 and int(gap.max()) <= 1


def recover(name: str, base_bits: np.ndarray, edited_bits: np.ndarray, dtype: str) -> Structure:
    mask = base_bits != edited_bits
    changed = int(mask.sum())
    s = Structure(name, tuple(base_bits.shape), dtype, changed)
    if changed == 0 or not is_float(dtype):
        s.form = "unchanged" if changed == 0 else "non-float change"
        return s

    base = to_float(base_bits, dtype).astype(np.float64)
    edited = to_float(edited_bits, dtype).astype(np.float64)
    delta = edited - base
    s.relative = float(np.linalg.norm(delta) / max(np.linalg.norm(base), 1e-30))

    if base.ndim != 2:
        s.rows, s.cols = changed, base.shape[-1] if base.ndim else 1
        s.form = "rounding" if _one_ulp(base_bits, edited_bits, mask) else "vector change"
        return s

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    s.rows, s.cols = len(rows), len(cols)
    s.support = cols

    if _one_ulp(base_bits, edited_bits, mask):
        s.form = "rounding"
        return s
    if not edited[mask].any():
        s.form = "zeroed"
        return s

    scale = float((base * edited).sum() / max((base * base).sum(), 1e-30))
    if np.linalg.norm(edited - scale * base) <= 0.02 * np.linalg.norm(delta):
        s.form, s.alpha = "scaled", scale
        return s

    block = delta[np.ix_(rows, cols)]
    u, sv, vh = _top_singular(block)
    sigma = _rounding_sigma(edited[np.ix_(rows, cols)], dtype)
    s.rank = max(int((sv > 1.5 * sigma * (np.sqrt(block.shape[0]) + np.sqrt(block.shape[1]))).sum()), 1)
    s.energy = float(sv[0] ** 2 / (sv**2).sum())
    s.form = "rank"

    if s.rank == 1:
        top, right = u[:, 0], vh[0]
        restricted = base[np.ix_(rows, cols)]
        norm = np.linalg.norm(block)

        coef = top @ restricted
        out_alpha = -float(sv[0] * right @ coef / max(coef @ coef, 1e-30))
        out_res = float(np.linalg.norm(block + out_alpha * np.outer(top, coef)) / norm)

        image = restricted @ right
        in_alpha = -float(sv[0] * top @ image / max(image @ image, 1e-30))
        in_res = float(np.linalg.norm(block + in_alpha * np.outer(image, right)) / norm)

        # A fit is good when what it leaves behind is no larger than rounding explains.
        allowed = max(0.05, 1.5 * sigma * np.sqrt(block.size) / norm)
        if min(out_res, in_res) < allowed:
            s.form = "projection"
            if out_res <= in_res:
                s.side, s.alpha, s.residual = "output", out_alpha, out_res
                direction = np.zeros(base.shape[0])
                direction[rows] = top
            else:
                s.side, s.alpha, s.residual = "input", in_alpha, in_res
                direction = np.zeros(base.shape[1])
                direction[cols] = right
            s.direction = direction
    return s


def shared_directions(structures: list[Structure], threshold: float = 0.9) -> list[list[str]]:
    """Group projection edits that remove the same direction, across tensors."""
    items = [s for s in structures if s.direction is not None]
    groups: list[list[Structure]] = []
    for s in items:
        for g in groups:
            head = g[0].direction
            if head.shape == s.direction.shape and abs(head @ s.direction) >= threshold:
                g.append(s)
                break
        else:
            groups.append([s])
    return [[s.name for s in g] for g in groups if len(g) > 1]
