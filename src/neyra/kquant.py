"""A numpy Q4_K quantizer written from llama.cpp's reference algorithm (quantize_row_q4_K_ref, without an
importance matrix), so the stored bytes of a K-quant file can be rebuilt from the weights it came from.

gguf-py quantizes only the legacy types. This follows ggml-quants.c's make_qkx2_quants step for step and keeps
float32 arithmetic in the same order, one element after another. A C compiler may fuse a multiply and an add into a
single rounding on some CPUs, so `fused` offers that variant; which one a file was made with is found by trying
both, and a quantization that matches neither is not claimed.
"""

from __future__ import annotations

import numpy as np

F32 = np.float32
BLOCK, SUB = 256, 32


def nearest(x: np.ndarray) -> np.ndarray:
    """ggml's nearest_int: round half to even."""
    return np.rint(x).astype(np.int32)


def mad(a, b, c, fused: bool):
    """a * b + c in float32, with one rounding when fused."""
    if fused:
        return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(F32)
    return (a * b).astype(F32) + c


def make_qkx2(x: np.ndarray, w: np.ndarray, nmax: int, rmin: float, rdelta: float, nstep: int, fused: bool):
    """make_qkx2_quants for many sub-blocks at once. x and w are float32 with shape (m, 32)."""
    m, n = x.shape
    lo, hi = x.min(axis=1), x.max(axis=1)
    sum_w = w[:, 0].copy()
    sum_x = (sum_w * x[:, 0]).astype(F32)
    for i in range(1, n):
        sum_w = sum_w + w[:, i]
        sum_x = mad(w[:, i], x[:, i], sum_x, fused)
    lo = np.minimum(lo, F32(0))
    flat = hi == lo
    span = np.where(flat, F32(1), hi - lo).astype(F32)
    iscale = (F32(nmax) / span).astype(F32)
    scale = (F32(1) / iscale).astype(F32)
    levels = np.clip(nearest((iscale[:, None] * (x - lo[:, None])).astype(F32)), 0, nmax)
    best = np.zeros(m, F32)
    for i in range(n):
        diff = (mad(scale, levels[:, i].astype(F32), lo, fused) - x[:, i]).astype(F32)
        best = mad(w[:, i], (diff * diff).astype(F32), best, fused)
    keep = levels.copy()
    for step in range(nstep + 1):
        iscale = ((F32(rmin) + F32(rdelta) * F32(step) + F32(nmax)).astype(F32) / span).astype(F32)
        aux = np.clip(nearest((iscale[:, None] * (x - lo[:, None])).astype(F32)), 0, nmax)
        sum_l = np.zeros(m, F32)
        sum_l2 = np.zeros(m, F32)
        sum_xl = np.zeros(m, F32)
        for i in range(n):
            lf = aux[:, i].astype(F32)
            wl = (w[:, i] * lf).astype(F32)
            sum_l = sum_l + wl
            sum_l2 = mad(wl, lf, sum_l2, fused)
            sum_xl = mad(wl, x[:, i], sum_xl, fused)
        d = (sum_w * sum_l2 - (sum_l * sum_l).astype(F32)).astype(F32)
        ok = d > 0
        safe = np.where(ok, d, F32(1)).astype(F32)
        this_scale = (((sum_w * sum_xl).astype(F32) - (sum_x * sum_l).astype(F32)).astype(F32) / safe).astype(F32)
        this_min = (((sum_l2 * sum_x).astype(F32) - (sum_l * sum_xl).astype(F32)).astype(F32) / safe).astype(F32)
        up = this_min > 0
        this_min = np.where(up, F32(0), this_min).astype(F32)
        this_scale = np.where(up, (sum_xl / np.where(sum_l2 == 0, F32(1), sum_l2)).astype(F32), this_scale).astype(F32)
        err = np.zeros(m, F32)
        for i in range(n):
            diff = (mad(this_scale, aux[:, i].astype(F32), this_min, fused) - x[:, i]).astype(F32)
            err = mad(w[:, i], (diff * diff).astype(F32), err, fused)
        better = ok & (err < best)
        keep = np.where(better[:, None], aux, keep)
        best = np.where(better, err, best)
        scale = np.where(better, this_scale, scale).astype(F32)
        lo = np.where(better, this_min, lo).astype(F32)
    keep = np.where(flat[:, None], 0, keep)
    scale = np.where(flat, F32(0), scale).astype(F32)
    return scale, (-lo).astype(F32), keep


def quantize_q4_k(values: np.ndarray, fused: bool = False) -> np.ndarray:
    """Q4_K blocks (144 bytes per 256 values) for float32 values whose last axis is a multiple of 256."""
    flat = np.ascontiguousarray(values, dtype=F32).reshape(-1, BLOCK)
    nb = len(flat)
    x = flat.reshape(nb * 8, SUB)
    sq = np.zeros(nb * 8, F32)
    for i in range(SUB):
        sq = mad(x[:, i], x[:, i], sq, fused)
    av = np.sqrt((sq / F32(SUB)).astype(F32)).astype(F32)
    weights = (av[:, None] + np.abs(x)).astype(F32)
    scales, mins, levels = make_qkx2(x, weights, 15, -1.0, 0.1, 20, fused)
    scales, mins = scales.reshape(nb, 8), mins.reshape(nb, 8)
    max_scale, max_min = scales.max(axis=1), mins.max(axis=1)
    inv_scale = np.where(max_scale > 0, F32(63) / np.where(max_scale > 0, max_scale, F32(1)), F32(0)).astype(F32)
    inv_min = np.where(max_min > 0, F32(63) / np.where(max_min > 0, max_min, F32(1)), F32(0)).astype(F32)
    ls = np.minimum(63, nearest((inv_scale[:, None] * scales).astype(F32)))
    lm = np.minimum(63, nearest((inv_min[:, None] * mins).astype(F32)))
    packed = np.zeros((nb, 12), np.uint8)
    packed[:, :4] = ls[:, :4] | ((ls[:, 4:] >> 4) << 6)
    packed[:, 4:8] = lm[:, :4] | ((lm[:, 4:] >> 4) << 6)
    packed[:, 8:12] = (ls[:, 4:] & 0xF) | ((lm[:, 4:] & 0xF) << 4)
    d16 = (max_scale / F32(63)).astype(F32).astype(np.float16)
    m16 = (max_min / F32(63)).astype(F32).astype(np.float16)
    d, dmin = d16.astype(F32), m16.astype(F32)
    levels = levels.reshape(nb, 8, SUB)
    step = (d[:, None] * ls.astype(F32)).astype(F32)
    off = (dmin[:, None] * lm.astype(F32)).astype(F32)
    live = step != 0
    safe = np.where(live, step, F32(1)).astype(F32)
    again = np.clip(nearest((((x.reshape(nb, 8, SUB) + off[:, :, None]).astype(F32)) / safe[:, :, None]).astype(F32)), 0, 15)
    levels = np.where(live[:, :, None], again, levels).astype(np.uint8)
    pairs = levels.reshape(nb, 4, 2, SUB)
    qs = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).reshape(nb, 128).astype(np.uint8)
    out = np.empty((nb, 144), np.uint8)
    out[:, 0:2] = d16.reshape(nb, 1).view(np.uint8)
    out[:, 2:4] = m16.reshape(nb, 1).view(np.uint8)
    out[:, 4:16] = packed
    out[:, 16:] = qs
    return out
