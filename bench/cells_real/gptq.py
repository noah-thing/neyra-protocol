"""GPTQ and AWQ files against their own true source: cells should refute them (not nearest rounding of the source).

usage: python -m bench.cells_real.gptq GPTQ_OR_AWQ_REPO TRUE_REPO gptq|awq
"""
import json
import sys

import numpy as np

from .lib import Safetensors, bf16_to_f32, get

SLACK = 2.0**-13
ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def unpack_cols(q):  # (a, b) int32 -> (a, b*8) nibbles in storage order
    q = q.view(np.uint32)
    return np.stack([(q >> (4 * p)) & 15 for p in range(8)], axis=-1).reshape(q.shape[0], q.shape[1] * 8)


def unpack_rows(q):  # (a, b) -> (a*8, b)
    q = q.view(np.uint32)
    return np.stack([(q >> (4 * p)) & 15 for p in range(8)], axis=1).reshape(q.shape[0] * 8, q.shape[1])


def load(st, name):
    m = st.meta[name]
    s, e = m["data_offsets"]
    raw = np.frombuffer(get(st.url, st.base + s, st.base + e - 1), np.uint8)
    dt = {"I32": np.int32, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}[m["dtype"]]
    return raw.view(dt).reshape(m["shape"])


def main():
    repo, true_repo, mode = sys.argv[1:4]
    q = Safetensors(repo)
    truth = Safetensors(true_repo)
    out = {"repo": repo, "mode": mode, "tensors": 0, "refuted": 0, "live": 0, "outside": 0, "cos": [], "overshoot_p99": []}
    bases = sorted({n[:-len(".qweight")] for n in q.meta if n.endswith(".qweight")})
    for base in bases:
        w_true = truth.read(base + ".weight")[0]
        w_true = bf16_to_f32(w_true) if w_true.dtype == np.uint16 else w_true.astype(np.float32)  # (out, in)
        qw, qz, sc = load(q, base + ".qweight"), load(q, base + ".qzeros"), load(q, base + ".scales").astype(np.float32)
        if mode == "gptq":
            codes = unpack_rows(qw).astype(np.int32)  # (in, out)
            zeros = unpack_cols(qz).astype(np.int32)  # (groups, out)
            gidx = load(q, base + ".g_idx") if base + ".g_idx" in q.meta else np.arange(codes.shape[0]) // (codes.shape[0] // sc.shape[0])
        else:
            n, c = qw.shape[0], qw.shape[1]
            raw = unpack_cols(qw).reshape(n, c, 8)
            codes = np.empty((n, c * 8), np.int32)
            for p in range(8):
                codes[:, np.arange(c) * 8 + ORDER[p]] = raw[:, :, p]
            rz = unpack_cols(qz).reshape(qz.shape[0], qz.shape[1], 8)
            zeros = np.empty((qz.shape[0], qz.shape[1] * 8), np.int32)
            for p in range(8):
                zeros[:, np.arange(qz.shape[1]) * 8 + ORDER[p]] = rz[:, :, p]
            gidx = np.arange(codes.shape[0]) // (codes.shape[0] // sc.shape[0])
        best = None
        for shift in ((1, 0) if mode == "gptq" else (0,)):
            z = (zeros + shift)[gidx]  # (in, out)
            s = sc[gidx]
            centre = ((codes - z) * s).T  # (out, in)
            cos = float((centre * w_true).sum() / np.sqrt((centre ** 2).sum() * (w_true ** 2).sum()))
            if best is None or cos > best[0]:
                best = (cos, centre, s.T, codes.T)
        cos, centre, step, c = best
        step = np.abs(step)
        half = step * (0.5 + SLACK)
        low = np.where(c == 0, -np.inf, centre - half)
        high = np.where(c == 15, np.inf, centre + half)
        live = step > 0
        bad = live & ((w_true < low) | (w_true > high))
        over = np.where(bad, np.maximum(low - w_true, w_true - high) / np.where(step > 0, step, 1), 0)
        out["tensors"] += 1
        out["refuted"] += int(bad.any())
        out["live"] += int(live.sum())
        out["outside"] += int(bad.sum())
        out["cos"].append(round(cos, 4))
        out["overshoot_p99"].append(float(np.percentile(over[bad], 99)) if bad.any() else 0.0)
    out["cos_min"] = min(out["cos"])
    out["cos"] = out["cos"][:6]
    out["overshoot_p99_max"] = max(out["overshoot_p99"])
    out.pop("overshoot_p99")
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
