"""Where a quantized weight's source must have been.

A block quantizer picks a scale, then stores the nearest code to each weight divided by that scale. So a stored
code is a certificate about its source: the weight sat inside the code's cell, an interval around the dequantized
value. Decoding a file and checking a candidate against the cells needs no quantizer run, and does not depend on how
the scales were searched for (importance matrix, different llama.cpp version): the scales are in the file.

A cell is a necessary condition. A candidate that falls outside one cell was not the input of the quantizer, whatever
else is true. A candidate inside every cell may be the source or may be any of the many other tensors that round to
the same codes; "consistent with" is the strongest thing this module can say.

Each quantizer computes the code in float32, so every cell is widened by SLACK of its step. Q8_0 picks the code with
the unrounded scale and stores it rounded to float16, which widens its cells by the rounding error of the scale per
unit of code. That error is absolute once the scale is below 2^-14, which a block of small weights reaches: the first
version of this module missed it and refuted the true source in 15 of 197 tensors of a real Q8_0 file.

The legacy types (Q4_0, Q4_1, Q5_0, Q5_1) are left out on purpose. With an importance matrix their codes come from
a weighted search for the scale, not from rounding to the stored one, and cells built on that assumption refuted
sources that were true (bench/cells_vs_llamacpp.py shows it, with the llama.cpp source compiled). Without an
importance matrix they would hold, but a file does not say which it was.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (block width in elements, bytes per block)
LAYOUT = {"Q8_0": (32, 34), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210)}
SLACK = 2.0**-13
F16_HALF_ULP = 2.0**-11 * 1.01
F16_SUBNORMAL_HALF_ULP = 2.0**-25 * 1.01


@dataclass(frozen=True)
class Cells:
    centre: np.ndarray  # float32, what a reader dequantizes
    low: np.ndarray  # float64, -inf where the code is clamped at the bottom
    high: np.ndarray
    live: np.ndarray  # False where the scale is zero and the code says nothing about the source


def supported(kind: str) -> bool:
    return kind in LAYOUT


def _blocks(kind: str, raw: np.ndarray) -> np.ndarray:
    if kind not in LAYOUT:
        raise ValueError(f"no cell model for {kind}")
    width = LAYOUT[kind][1]
    flat = np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    if flat.size % width:
        raise ValueError(f"{kind} data is {flat.size} bytes, not a whole number of {width}-byte blocks")
    return flat.reshape(-1, width)


def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.float16).astype(np.float32).reshape(-1)


def _scale_min(sc12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """get_scale_min_k4: eight 6-bit scales and eight 6-bit mins packed in 12 bytes."""
    sc = np.empty((sc12.shape[0], 8), np.uint8)
    mn = np.empty_like(sc)
    sc[:, :4] = sc12[:, :4] & 63
    mn[:, :4] = sc12[:, 4:8] & 63
    sc[:, 4:] = (sc12[:, 8:12] & 0xF) | ((sc12[:, :4] >> 6) << 4)
    mn[:, 4:] = (sc12[:, 8:12] >> 4) | ((sc12[:, 4:8] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def _interval(centre, step, q, top, bottom=0, extra=0.0):
    """Cell of code q for a quantizer whose code is round((x - centre_of_zero) / step), clamped to [bottom, top]."""
    half = np.abs(step) * (0.5 + SLACK) + extra
    low = np.where(q == bottom, -np.inf, centre - half)
    high = np.where(q == top, np.inf, centre + half)
    return low, high


def decode(kind: str, raw: np.ndarray) -> Cells:
    """Dequantized values and cells for every element of a whole number of blocks, in storage order."""
    b = _blocks(kind, raw)
    nb = len(b)
    if kind in ("Q4_K", "Q5_K"):
        d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
        sc, mn = _scale_min(b[:, 4:16])
        qs = b[:, 16:] if kind == "Q4_K" else b[:, 48:]
        q = np.empty((nb, 4, 2, 32), np.int32)
        q[:, :, 0] = (qs.reshape(nb, 4, 32) & 0xF)
        q[:, :, 1] = (qs.reshape(nb, 4, 32) >> 4)
        top = 15
        if kind == "Q5_K":
            qh = b[:, 16:48].astype(np.int32)
            for g in range(4):
                q[:, g, 0] += ((qh >> (2 * g)) & 1) << 4
                q[:, g, 1] += ((qh >> (2 * g + 1)) & 1) << 4
            top = 31
        q = q.reshape(nb, 8, 32)
        step = (d[:, None] * sc)[:, :, None]
        off = (dmin[:, None] * mn)[:, :, None]
        centre = (step * q.astype(np.float32) - off).astype(np.float32)
        s64 = step.astype(np.float64)
        low, high = _interval(centre.astype(np.float64), s64, q, top)
        live = np.broadcast_to(s64 > 0, q.shape)
        return _flat(centre, low, high, live)
    if kind == "Q6_K":
        ql, qh = b[:, :128], b[:, 128:192]
        sc = b[:, 192:208].view(np.int8).astype(np.float32)
        d = _f16(b[:, 208:210])
        q = np.empty((nb, 2, 4, 32), np.int32)
        a, c = ql.reshape(nb, 2, 2, 32), qh.reshape(nb, 2, 32).astype(np.int32)
        q[:, :, 0] = (a[:, :, 0] & 0xF) | ((c & 3) << 4)
        q[:, :, 1] = (a[:, :, 1] & 0xF) | (((c >> 2) & 3) << 4)
        q[:, :, 2] = (a[:, :, 0] >> 4) | (((c >> 4) & 3) << 4)
        q[:, :, 3] = (a[:, :, 1] >> 4) | (((c >> 6) & 3) << 4)
        # sixteen scales per block, one per run of 16 elements
        step = (d[:, None] * sc).reshape(nb, 2, 4, 2)
        step = np.repeat(step, 16, axis=3)
        centre = (step * (q - 32).astype(np.float32)).astype(np.float32)
        s64 = step.astype(np.float64)
        half = np.abs(s64) * (0.5 + SLACK)
        c64 = centre.astype(np.float64)
        neg = s64 < 0
        # a negative scale reverses which end of the code range is unbounded
        low = np.where(np.where(neg, q == 63, q == 0), -np.inf, c64 - half)
        high = np.where(np.where(neg, q == 0, q == 63), np.inf, c64 + half)
        return _flat(centre, low, high, np.broadcast_to(s64 != 0, q.shape))
    if kind == "Q8_0":
        d = _f16(b[:, 0:2])[:, None]
        q = b[:, 2:].view(np.int8).astype(np.int32)
        centre = (d * q.astype(np.float32)).astype(np.float32)
        step = np.abs(d).astype(np.float64)
        # The code is picked with the unrounded scale and the float16 one is stored. Below 2^-14 float16 has
        # a fixed spacing, so the error of the stored scale stops shrinking with it.
        scale_error = np.maximum(step * F16_HALF_ULP, F16_SUBNORMAL_HALF_ULP)
        low, high = _interval(centre.astype(np.float64), step, q, 127, -128, np.abs(q) * scale_error)
        return _flat(centre, low, high, np.broadcast_to(step > 0, q.shape))
    raise ValueError(f"no cell model for {kind}")


def _flat(*arrays) -> Cells:
    return Cells(*(np.ascontiguousarray(a).reshape(-1) for a in arrays))


def violations(cells: Cells, x: np.ndarray) -> np.ndarray:
    """Indices of elements whose candidate source value lies outside the cell."""
    x = np.asarray(x, np.float64).reshape(-1)
    out = cells.live & ((x < cells.low) | (x > cells.high))
    return np.flatnonzero(out)


@dataclass(frozen=True)
class Fit:
    outside: int  # live elements whose candidate lies outside its cell
    live: int
    first: int | None  # flat index of the first such element
    route: str


ROUTES = ("direct", "via F16", "via BF16")


def _through(x: np.ndarray, route: str) -> np.ndarray:
    if route == "via F16":
        return x.astype(np.float16).astype(np.float32)
    if route == "via BF16":
        bits = x.astype(np.float32).view(np.uint32).astype(np.uint64)
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return x


def fit(kind: str, raw: np.ndarray, x: np.ndarray, chunk_blocks: int = 8192) -> Fit:
    """Count the candidate's values outside the cells of the stored tensor, in chunks so a large tensor stays small.

    llama-quantize may have read the weights from a float16 or bfloat16 GGUF, so the candidate is also tried after
    rounding that way; the route with the fewest values outside is reported.
    """
    width = LAYOUT[kind][0]
    blocks = _blocks(kind, raw)
    flat = np.ascontiguousarray(x, np.float32).reshape(-1)
    if flat.size != len(blocks) * width:
        raise ValueError(f"{kind} tensor holds {len(blocks) * width} values, the candidate {flat.size}")
    best: Fit | None = None
    for route in ROUTES:
        outside, live, first = 0, 0, None
        for lo in range(0, len(blocks), chunk_blocks):
            part = blocks[lo:lo + chunk_blocks]
            cells = decode(kind, part)
            bad = violations(cells, _through(flat[lo * width:(lo + len(part)) * width], route))
            outside += len(bad)
            live += int(cells.live.sum())
            if first is None and len(bad):
                first = lo * width + int(bad[0])
        found = Fit(outside, live, first, route)
        if best is None or outside < best.outside:
            best = found
        if outside == 0:
            break
    return best
