"""Paired comparisons. Two variants answer the same items, so compare within items."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Paired:
    n: int
    base: float
    edited: float
    difference: float
    stderr: float
    low: float
    high: float
    gained: int = 0
    lost: int = 0
    p_value: float | None = None


def sign_test(gained: int, lost: int) -> float:
    """Two sided exact McNemar test on the items that changed."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(base: np.ndarray, edited: np.ndarray, boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Paired:
    base = np.asarray(base, dtype=np.float64)
    edited = np.asarray(edited, dtype=np.float64)
    if base.shape != edited.shape or base.ndim != 1 or len(base) == 0:
        raise ValueError("paired comparison needs two equal length vectors")
    d = edited - base
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (boot, len(d)))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    binary = set(np.unique(np.concatenate([base, edited]))) <= {0.0, 1.0}
    gained, lost = (int((d > 0).sum()), int((d < 0).sum())) if binary else (0, 0)
    return Paired(len(d), float(base.mean()), float(edited.mean()), float(d.mean()),
                  float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0,
                  float(low), float(high), gained, lost,
                  sign_test(gained, lost) if binary else None)


def kl_from_logits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) per row, in nats."""
    p = p - p.max(axis=-1, keepdims=True)
    q = q - q.max(axis=-1, keepdims=True)
    lp = p - np.log(np.exp(p).sum(axis=-1, keepdims=True))
    lq = q - np.log(np.exp(q).sum(axis=-1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(axis=-1)
