"""The dormant clock: an exact claim about rows that training never fed, bound to a rows root.

Embedding rows for tokens that never occur get no gradient. When the optimizer only decays them, a
later checkpoint holds b = round(c * a) for one scalar c, row after row. The test below asks whether
any c makes every entry of b the correctly rounded c * a. The answer is exact: it intersects the
rounding cell of each entry as an interval of c, with no tolerance. Each side's rows come with a row
proof, so the claim is tied to the rows roots of the two checkpoints by about 2 MB of reads.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np

from .protocol import ProtocolError, digest, jcs
from .rows import fetch, parse_ranges, prove, rows_array, verify_proof
from .store import Checkpoint

TYPE = "neyra.dormant_scale/1"
MIN_N, MIN_DISTINCT = 10_000, 256
FORMAT = {"BF16": (8, 7), "F16": (5, 10), "F32": (8, 23)}
EMBEDDINGS = (r"embed_tokens\.weight$", r"embed_in\.weight$", r"wte\.weight$", r"word_embeddings\.weight$")


def decompose(bits: np.ndarray, dtype: str):
    """sign, integer mantissa m and exponent e with value = sign * m * 2**e, and two flags used by cells."""
    ebits, mbits = FORMAT[dtype]
    bias = (1 << (ebits - 1)) - 1
    b = bits.astype(np.int64)
    sign = (b >> (ebits + mbits)) & 1
    exp = (b >> mbits) & ((1 << ebits) - 1)
    frac = b & ((1 << mbits) - 1)
    normal = exp > 0
    m = np.where(normal, frac | (1 << mbits), frac)
    e = np.where(normal, exp - bias - mbits, 1 - bias - mbits)
    return sign, m, e, normal & (frac == 0) & (exp > 1), (frac & 1) == 0


def cells(bits: np.ndarray, dtype: str):
    """Bounds of the interval that rounds to each magnitude, and whether its ends round to it (even mantissa)."""
    sign, m, e, pow2, even = decompose(bits, dtype)
    lo = np.ldexp(np.where(pow2, 2 * m - 0.5, 2 * m - 1.0) / 2.0, e)
    hi = np.ldexp((2 * m + 1.0) / 2.0, e)
    return sign, lo, hi, even


def interval(a: np.ndarray, b: np.ndarray, dtype: str):
    """Per entry, the c with b = round(c * a) is in [l, h]."""
    sa, ma, ea, _, _ = decompose(a, dtype)
    sb, lo, hi, even = cells(b, dtype)
    mag = np.ldexp(ma.astype(np.float64), ea)
    with np.errstate(divide="ignore", invalid="ignore"):
        return lo / mag, hi / mag, even, sa == sb, ma == 0


def exact_bounds(a: np.ndarray, b: np.ndarray, dtype: str, i: int):
    """Entry i's bounds on c as exact fractions, with whether the upper and lower ends are attained."""
    _, ma, ea, _, _ = decompose(a[i:i + 1], dtype)
    _, mb, eb, pow2, even = decompose(b[i:i + 1], dtype)
    mag = Fraction(int(ma[0])) * Fraction(2) ** int(ea[0])
    unit = Fraction(2) ** int(eb[0])
    below = Fraction(1, 4) if pow2[0] else Fraction(1, 2)
    return (int(mb[0]) - below) * unit / mag, (int(mb[0]) + Fraction(1, 2)) * unit / mag, bool(even[0])


def exact_test(a_bits: np.ndarray, b_bits: np.ndarray, dtype: str, min_n: int = MIN_N,
               min_distinct: int = MIN_DISTINCT) -> dict:
    a, b = a_bits.reshape(-1), b_bits.reshape(-1)
    r: dict = {"dtype": dtype, "n": int(a.size)}
    if np.array_equal(a, b):
        return {**r, "verdict": "IDENTICAL", "n_nonzero": int((a != 0).sum())}
    low, high, _, sign_ok, a_zero = interval(a, b, dtype)
    b_zero = (b & ((1 << sum(FORMAT[dtype])) - 1)) == 0
    use = ~a_zero
    r["n_nonzero"] = int(use.sum())
    r["a_zero_b_nonzero"] = int((a_zero & ~b_zero).sum())
    r["distinct_a"] = len(np.unique(a[use])) if use.any() else 0
    r["sign_flips"] = int((use & ~sign_ok).sum())
    if not use.any():
        return {**r, "verdict": "UNDECIDED"}
    at = np.flatnonzero(use)
    lo_f, _, lo_closed = exact_bounds(a, b, dtype, int(at[np.argmax(low[use])]))
    _, hi_f, hi_closed = exact_bounds(a, b, dtype, int(at[np.argmin(high[use])]))
    r["c_lo"], r["c_hi"] = float(lo_f), float(hi_f)
    r["c_lo_frac"], r["c_hi_frac"] = str(lo_f), str(hi_f)
    r["viol_ext"] = r["a_zero_b_nonzero"] + r["sign_flips"]
    nonempty = lo_f < hi_f or (lo_f == hi_f and lo_closed and hi_closed)
    fit = bool(nonempty and r["viol_ext"] == 0 and hi_f > 0)
    ok = use & sign_ok
    if not ok.any():
        return {**r, "verdict": "REJECT", "min_violations": int(use.sum()), "best_cover": 0, "n_checked": 0}
    # the most entries any one c satisfies, by sweeping the interval ends
    ends = np.concatenate([np.stack([low[ok], np.ones(ok.sum())], 1), np.stack([high[ok], -np.ones(ok.sum())], 1)])
    ends = ends[np.lexsort((-ends[:, 1], ends[:, 0]))]
    cover = np.cumsum(ends[:, 1])
    r["best_cover"], r["n_checked"] = int(cover.max()), int(ok.sum())
    r["min_violations"] = int(ok.sum() - cover.max()) + r["viol_ext"]
    enough = r["n_nonzero"] >= min_n and r["distinct_a"] >= min_distinct
    r["verdict"] = ("FIT" if fit else "REJECT") if enough else ("UNDECIDED_FIT" if fit else "REJECT_SMALL")
    return r


def common_master_test(a_bits: np.ndarray, b_bits: np.ndarray, dtype: str, min_n: int = MIN_N,
                       min_distinct: int = MIN_DISTINCT) -> dict:
    """Both are roundings of one master w with unknown scalars. Feasible c = s2 / s1 is exact interval intersection."""
    a, b = a_bits.reshape(-1), b_bits.reshape(-1)
    sa, la, ha, _ = cells(a, dtype)
    sb, lb, hb, _ = cells(b, dtype)
    a_zero, b_zero = decompose(a, dtype)[1] == 0, decompose(b, dtype)[1] == 0
    live = ~a_zero & ~b_zero
    r = {"n_nonzero": int(live.sum()), "sign_flips": int((live & (sa != sb)).sum()),
         "zero_mismatch": int((a_zero != b_zero).sum())}
    if not live.any():
        return {**r, "verdict": "UNDECIDED"}
    with np.errstate(divide="ignore", invalid="ignore"):
        low, high = (lb / ha)[live], (hb / la)[live]
    r["c_lo"], r["c_hi"] = float(low.max()), float(high.min())
    r["distinct_a"] = len(np.unique(a[live]))
    ends = np.concatenate([np.stack([low, np.ones_like(low)], 1), np.stack([high, -np.ones_like(high)], 1)])
    ends = ends[np.lexsort((-ends[:, 1], ends[:, 0]))]
    r["min_violations"] = int(live.sum() - np.cumsum(ends[:, 1]).max()) + r["sign_flips"] + r["zero_mismatch"]
    fit = r["c_lo"] <= r["c_hi"] and r["sign_flips"] == 0 and r["zero_mismatch"] == 0 and r["c_hi"] > 0
    if r["n_nonzero"] >= min_n and r["distinct_a"] >= min_distinct:
        r["verdict"] = "FIT" if fit else "REJECT"
    else:
        r["verdict"] = "UNDECIDED_FIT" if fit else "REJECT_SMALL"
    return r


def decimal(x: float, places: int = 6) -> str:
    return f"{float(x):.{places}f}"


def analyse(a: np.ndarray, b: np.ndarray, dtype: str) -> dict:
    if a.size == 0:
        return {"n": 0, "distinct_a": 0, "verdict": "undecided"}
    forward, backward = exact_test(a, b, dtype), exact_test(b, a, dtype)
    master = common_master_test(a, b, dtype)
    out = {"n": int(a.size), "distinct_a": len(np.unique(a))}
    if forward["verdict"] == "IDENTICAL":
        out["verdict"] = "identical"
    elif forward["verdict"] == "FIT":
        lo, hi = Fraction(forward["c_lo_frac"]), Fraction(forward["c_hi_frac"])
        out.update(verdict="fit", c_lo=f"{lo.numerator}/{lo.denominator}", c_hi=f"{hi.numerator}/{hi.denominator}",
                   c_lo_dec=decimal(lo, 8), c_hi_dec=decimal(hi, 8),
                   d_lo=decimal(-math.log(float(hi)), 8) if hi > 0 else "inf", d_hi=decimal(-math.log(float(lo)), 8),
                   direction="a_before_b" if hi < 1 else ("b_before_a" if lo > 1 else "unordered"),
                   reverse_strict=backward["verdict"].lower())
    elif master["verdict"] == "FIT":
        out.update(verdict="common_master", c_lo_dec=decimal(master["c_lo"], 8), c_hi_dec=decimal(master["c_hi"], 8),
                   direction="unordered")
    elif (forward["verdict"].startswith("UNDECIDED") or master["verdict"].startswith("UNDECIDED")
          or forward["verdict"] == "REJECT_SMALL"):
        out["verdict"] = "undecided"
    else:
        out.update(verdict="reject", min_violations=int(forward.get("min_violations", -1)),
                   n_checked=int(forward.get("n_checked", -1)))
    return out


def find_embedding(ck: Checkpoint) -> str:
    for pattern in EMBEDDINGS:
        found = sorted((n for n in ck.tensors if re.search(pattern, n)), key=len)
        if found:
            return found[0]
    raise ProtocolError("bad_claim", "no embedding tensor found; name one with --tensor")


def read_rows(ck: Checkpoint, index, name: str, ranges: list[tuple[int, int]]) -> tuple[np.ndarray, dict]:
    """The rows and a proof for them. The proof is checked here, so a wrong index fails before any claim."""
    ref = ck.tensors[name]
    proof = prove(index, name, ranges)
    blocks = verify_proof(proof, fetch(ref, proof))
    return np.concatenate([rows_array(ref.dtype, ref.shape, blk) for blk in blocks]), proof


def make_claim(a_ck: Checkpoint, b_ck: Checkpoint, a_index, b_index, tensor: str | None, spec: str) -> dict:
    name = tensor or find_embedding(a_ck)
    ra, rb = a_ck.tensors.get(name), b_ck.tensors.get(name)
    if ra is None or rb is None or (ra.dtype, ra.shape) != (rb.dtype, rb.shape):
        raise ProtocolError("bad_claim", f"{name} is missing or differs in dtype or shape")
    if ra.dtype not in FORMAT or len(ra.shape) != 2:
        raise ProtocolError("bad_claim", f"{ra.dtype} {list(ra.shape)} is not a bf16, f16 or f32 matrix")
    ranges = parse_ranges(spec)
    a, a_proof = read_rows(a_ck, a_index, name, ranges)
    b, b_proof = read_rows(b_ck, b_index, name, ranges)
    claim = {"type": TYPE, "tensor": name, "dtype": ra.dtype, "shape": list(ra.shape),
             "rows": [list(r) for r in ranges], "a_rows_root": a_index.rows_root, "b_rows_root": b_index.rows_root,
             "a_proof": a_proof, "b_proof": b_proof, "guards": {"min_n": MIN_N, "min_distinct": MIN_DISTINCT},
             **analyse(a, b, ra.dtype)}
    claim["id"] = digest("payload", jcs(claim))
    return claim


def check_claim(claim: dict, a_ck: Checkpoint, b_ck: Checkpoint) -> list[str]:
    """Every way the claim fails to follow from the bytes. Empty means it holds."""
    body = {k: v for k, v in claim.items() if k != "id"}
    if claim.get("id") != digest("payload", jcs(body)):
        return ["the id does not match the content"]
    sides = {}
    for side, ck in (("a", a_ck), ("b", b_ck)):
        proof = claim[f"{side}_proof"]
        if proof["rows_root"] != claim[f"{side}_rows_root"] or proof["ranges"] != claim["rows"]:
            return [f"{side}: the proof is not for the rows this claim names"]
        if proof["tensor"] != {"name": claim["tensor"], "dtype": claim["dtype"], "shape": claim["shape"]}:
            return [f"{side}: the proof is for another tensor"]
        ref = ck.tensors.get(claim["tensor"])
        if ref is None:
            return [f"{side}: no tensor {claim['tensor']}"]
        blocks = verify_proof(proof, fetch(ref, proof))
        sides[side] = np.concatenate([rows_array(ref.dtype, ref.shape, blk) for blk in blocks])
    fresh = analyse(sides["a"], sides["b"], claim["dtype"])
    return [f"{k} is {claim.get(k)!r}, the rows give {v!r}" for k, v in fresh.items() if claim.get(k) != v]
