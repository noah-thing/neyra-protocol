"""How well does diff recover known edits? Plant them in real weights and score it.

Tensors come from a real checkpoint, so singular spectra and bf16 rounding are realistic.
Every edit is applied in float64 and stored in the tensor's dtype, as a real tool would.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from neyra.numerics import from_float, to_float
from neyra.recover import recover
from neyra.store import Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17")
ap.add_argument("-n", type=int, default=20, help="trials per condition")
ap.add_argument("-o", "--out", default="benchmarks/forensics.json")
a = ap.parse_args()

ck = Checkpoint.open(a.model)
wanted = [f"layers.{i}.mlp.down_proj.weight" for i in (2, 11, 19)] + ["layers.7.self_attn.o_proj.weight", "layers.5.linear_attn.out_proj.weight"]
names = [n for n in ck.tensors if any(n.endswith(w) for w in wanted) and "language_model" in n]
tensors = {n: ck.tensors[n].read() for n in names}
print(len(tensors), "tensors", file=sys.stderr, flush=True)
rng = np.random.default_rng(0)


def unit(n):
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def col_projection(w, k, alpha):
    u = unit(w.shape[0])
    cols = np.sort(rng.choice(w.shape[1], k, replace=False))
    out = w.copy()
    out[:, cols] -= alpha * np.outer(u, u @ w[:, cols])
    return out, {"form": "projection", "u": u, "cols": cols, "alpha": alpha}


def writer_projection(w, k, alpha):
    """Direction chosen to overlap W, as a real behavior direction does."""
    u = np.linalg.svd(w, full_matrices=False)[0][:, rng.integers(0, 32)]
    cols = np.sort(np.argsort(-np.abs(u @ w))[:k])
    out = w.copy()
    out[:, cols] -= alpha * np.outer(u, u @ w[:, cols])
    return out, {"form": "projection", "u": u, "cols": cols, "alpha": alpha}


def input_projection(w, alpha):
    v = unit(w.shape[1])
    return w - alpha * np.outer(w @ v, v), {"form": "projection", "alpha": alpha}


def additive(w, rank, scale):
    a, b = rng.standard_normal((w.shape[0], rank)), rng.standard_normal((rank, w.shape[1]))
    d = a @ b
    return w + scale * np.linalg.norm(w) / np.linalg.norm(d) * d, {"form": "rank", "rank": rank}


def zero_block(w, k):
    start = rng.integers(0, w.shape[1] - k)
    out = w.copy()
    out[:, start:start + k] = 0
    return out, {"form": "zeroed", "cols": np.arange(start, start + k)}


def scaled(w, c):
    return w * c, {"form": "scaled", "alpha": c}


CONDITIONS = {
    "writer projection k=64 a=6": lambda w: writer_projection(w, 64, 6.0),
    "writer projection k=64 a=1": lambda w: writer_projection(w, 64, 1.0),
    "writer projection k=8 a=2": lambda w: writer_projection(w, 8, 2.0),
    "writer projection k=512 a=1": lambda w: writer_projection(w, 512, 1.0),
    "random projection k=64 a=6": lambda w: col_projection(w, 64, 6.0),
    "random projection k=64 a=1": lambda w: col_projection(w, 64, 1.0),
    "random projection k=512 a=1": lambda w: col_projection(w, 512, 1.0),
    "full projection a=1": lambda w: col_projection(w, w.shape[1], 1.0),
    "input projection a=1": lambda w: input_projection(w, 1.0),
    "additive rank 1": lambda w: additive(w, 1, 0.05),
    "additive rank 2": lambda w: additive(w, 2, 0.05),
    "additive rank 8": lambda w: additive(w, 8, 0.1),
    "zeroed 256 columns": lambda w: zero_block(w, 256),
    "scaled 0.9": lambda w: scaled(w, 0.9),
}

rows = []
for label, fn in CONDITIONS.items():
    hits, cosines, alpha_err, f1s, ranks, got = 0, [], [], [], [], Counter()
    for t in range(a.n):
        name = names[t % len(names)]
        bits = tensors[name]
        dtype = ck.tensors[name].dtype
        w = to_float(bits, dtype).astype(np.float64)
        edited, truth = fn(w)
        s = recover(name, bits, from_float(edited.astype(np.float32), dtype).reshape(bits.shape), dtype)
        got[s.form] += 1
        hits += s.form == truth["form"]
        if "rank" in truth and s.rank is not None:
            ranks.append(s.rank == truth["rank"])
        if s.direction is not None and "u" in truth and len(s.direction) == len(truth["u"]):
            cosines.append(abs(s.direction @ truth["u"]))
        if s.alpha is not None and "alpha" in truth:
            alpha_err.append(abs(s.alpha - truth["alpha"]) / truth["alpha"])
        if "cols" in truth and s.support is not None:
            found, real = set(s.support.tolist()), set(truth["cols"].tolist())
            tp = len(found & real)
            f1s.append(2 * tp / max(len(found) + len(real), 1))
    row = {"condition": label, "n": a.n, "form_accuracy": hits / a.n, "forms": dict(got),
           "direction_cos": float(np.mean(cosines)) if cosines else None,
           "alpha_rel_error": float(np.median(alpha_err)) if alpha_err else None,
           "column_f1": float(np.mean(f1s)) if f1s else None,
           "rank_exact": float(np.mean(ranks)) if ranks else None}
    rows.append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

false_alarms = Counter()
for name, bits in tensors.items():
    dtype = ck.tensors[name].dtype
    w32 = to_float(bits, dtype)
    resaved = from_float(w32.copy(), dtype).reshape(bits.shape)
    false_alarms["resave_changed_bits"] += int(not np.array_equal(resaved, bits))
    for _ in range(a.n):
        noise = bits.copy().reshape(-1)
        picks = rng.choice(noise.size, noise.size // 100, replace=False)
        noise[picks] += rng.choice(np.array([1, 65535], dtype=np.uint16), len(picks))
        false_alarms[recover(name, bits, noise.reshape(bits.shape), dtype).form] += 1
print(json.dumps({"identity_and_one_ulp_noise": dict(false_alarms)}), flush=True)
Path(a.out).write_text(json.dumps({"model": ck.spec, "conditions": rows, "rounding_only": dict(false_alarms)}, indent=1))
