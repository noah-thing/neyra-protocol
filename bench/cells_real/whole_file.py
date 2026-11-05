"""Whole-file cell test of one public GGUF against several candidate sources, planted failures included.

usage: python -m bench.cells_real.whole_file GGUF_REPO FILE TRUE_REPO [name=repo ...]
Frozen protocol: neyra bench/cells_protocol.md. GGUF tensors are read by range and never stored.
"""
import json
import sys
import time
import zlib

import numpy as np

from neyra import cells

from .lib import HUB, QT, Safetensors, bf16_to_f32, gguf_header, gguf_raw, hf_name

COVERED = ("Q4_K", "Q5_K", "Q6_K", "Q8_0")
CHUNK = 1_500_000
ROUTES = cells.ROUTES


def rng_for(seed, *key):
    return np.random.default_rng([seed, *[zlib.crc32(k.encode()) if isinstance(k, str) else k for k in key]])


class Source:
    """One candidate: how to produce float32 rows of a tensor, whole-tensor state prepared once."""

    def __init__(self, label, st=None, mode="plain"):
        self.label, self.st, self.mode = label, st, mode

    def prepare(self, hf, truth, ti):
        self.hf, self.ti = hf, ti
        if self.mode == "unrelated":
            alt = hf
            if ".layers." in hf:
                i = int(hf.split(".layers.")[1].split(".")[0])
                alt = hf.replace(f".layers.{i}.", f".layers.{(i + 1) % self.layers}." if i + 1 < self.layers else ".layers.0.")
            self.arr = truth.read(alt)[0] if truth.has(alt) else truth.read(hf)[0]
            self.shift = 0 if truth.has(alt) and alt != hf else 1
        elif self.mode in ("lora", "ulp1", "step1"):
            self.arr = truth.read(hf)[0]
            if self.mode == "lora":
                total = 0.0
                for k in range(0, self.arr.shape[0], 4096):
                    a = np.asarray(self.arr[k:k + 4096])
                    w = bf16_to_f32(a) if a.dtype == np.uint16 else a.astype(np.float32)
                    total += float((w.astype(np.float64) ** 2).sum())
                self.rms = float(np.sqrt(total / self.arr.size))
                r = rng_for(11, hf)
                rank = 8
                self.A = r.standard_normal((self.arr.shape[0], rank)).astype(np.float32)
                self.B = r.standard_normal((rank, self.arr.shape[-1] if self.arr.ndim > 1 else 1)).astype(np.float32)
                self.gain = 0.002 * self.rms / np.sqrt(rank)
        else:
            self.arr = self.st.read(hf)[0]
        return self.arr.shape == truth.meta[hf]["shape"] if hasattr(truth, "meta") else True

    def rows(self, r0, r1, cell=None):
        if self.mode == "unrelated" and self.shift:
            a = np.asarray(self.arr[(np.arange(r0, r1) + 1) % self.arr.shape[0]])
        else:
            a = np.asarray(self.arr[r0:r1])
        x = bf16_to_f32(a) if a.dtype == np.uint16 else a.astype(np.float32)
        if self.mode == "lora":
            return x + (self.A[r0:r1] @ self.B) * np.float32(self.gain)
        if self.mode == "ulp1":
            m = rng_for(12, self.hf, r0).random(x.shape) < 0.01
            u = x.view(np.uint32)
            return np.where(m, u + np.uint32(1 << 16), u).view(np.float32)
        if self.mode == "step1":
            m = rng_for(13, self.hf, r0).random(x.size) < 0.01
            width = (cell.high - cell.low).reshape(x.shape)
            ok = np.isfinite(width) & m.reshape(x.shape) & cell.live.reshape(x.shape)
            return np.where(ok, x + width.astype(np.float32), x)
        return x


def run(gguf_repo, fname, true_repo, extra):
    t0 = time.time()
    url = HUB % (gguf_repo, fname)
    h = gguf_header(url)
    truth = Safetensors(true_repo)
    layers = h["kv"].get(next(k for k in h["kv"] if k.endswith(".block_count")))
    sources = [Source("true", truth)]
    for spec in extra:
        label, repo = spec.split("=")
        sources.append(Source(label, Safetensors(repo)))
    for mode in ("lora", "ulp1", "step1", "unrelated"):
        s = Source(mode, None, mode)
        s.layers = layers
        sources.append(s)
    census, cover = {}, {"covered_elems": 0, "uncovered_elems": 0, "uncovered_types": {}}
    res = {s.label: {"tensors": 0, "refuted": 0, "outside": 0, "live": 0, "first": None, "by_type": {}, "dot": 0.0, "nc": 0.0, "nx": 0.0, "nd": 0.0,
                         "floats_equal": 0, "floats_diff": 0, "missing": []} for s in sources}
    for gname, t in h["tensors"].items():
        kind = QT[t["type"]][0]
        census[kind] = census.get(kind, 0) + 1
        hf = hf_name(gname)
        n = int(np.prod(t["dims"]))
        if kind in COVERED and hf and truth.has(hf):
            cover["covered_elems"] += n
        else:
            cover["uncovered_elems"] += n
            cover["uncovered_types"][kind] = cover["uncovered_types"].get(kind, 0) + n
        if not hf or not truth.has(hf):
            continue
        shape = tuple(truth.meta[hf]["shape"])
        cols = t["dims"][0]
        rows = n // cols
        if kind not in COVERED:
            if kind in ("F32", "F16", "BF16"):
                raw = gguf_raw(h, gname)
                stored = {"F32": lambda b: np.frombuffer(b, np.float32), "F16": lambda b: np.frombuffer(b, np.float16).astype(np.float32),
                          "BF16": lambda b: bf16_to_f32(np.frombuffer(b, np.uint16))}[kind](raw)
                for s in sources:
                    if s.mode != "plain":
                        continue
                    x = s.st.read(hf)[0] if s.st.has(hf) else None
                    if x is None:
                        res[s.label]["missing"].append(hf)
                        continue
                    x = bf16_to_f32(x) if x.dtype == np.uint16 else x.astype(np.float32)
                    ok = x.reshape(-1).shape == stored.shape and any(
                        np.array_equal(cells._through(x.reshape(-1), r), stored) for r in ROUTES)
                    res[s.label]["floats_equal" if ok else "floats_diff"] += 1
            continue
        raw = np.frombuffer(gguf_raw(h, gname), np.uint8)
        rb = cols // QT[t["type"]][1] * QT[t["type"]][2]
        raw = raw.reshape(rows, rb)
        active = []
        for s in sources:
            try:
                if s.mode == "plain" and not s.st.has(hf):
                    res[s.label]["missing"].append(hf)
                    continue
                s.prepare(hf, truth, t)
                if tuple(s.arr.shape) != shape:
                    res[s.label]["missing"].append(hf + " shape")
                    continue
                active.append(s)
            except (ValueError, KeyError, OSError) as e:
                res[s.label]["missing"].append(f"{hf} {e}")
        per = {s.label: {"out": {r: 0 for r in ROUTES}, "live": 0, "first": None} for s in active}
        step = max(1, CHUNK // cols)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            c = cells.decode(kind, raw[r0:r1])
            for s in active:
                x = s.rows(r0, r1, c).reshape(-1)
                R = res[s.label]
                R["dot"] += float((c.centre.astype(np.float64) * x).sum())
                R["nc"] += float((c.centre.astype(np.float64) ** 2).sum())
                R["nx"] += float((x.astype(np.float64) ** 2).sum())
                R["nd"] += float(((c.centre.astype(np.float64) - x) ** 2).sum())
                P = per[s.label]
                P["live"] += int(c.live.sum())
                for route in ROUTES:
                    bad = cells.violations(c, cells._through(x, route))
                    P["out"][route] += len(bad)
                    if route == "direct" and len(bad) and P["first"] is None:
                        P["first"] = r0 * cols + int(bad[0])
        for s in active:
            P, R = per[s.label], res[s.label]
            best = min(P["out"].values())
            R["tensors"] += 1
            R["live"] += P["live"]
            R["outside"] += best
            T = R["by_type"].setdefault(kind, [0, 0, 0, 0])
            T[0] += 1
            T[2] += best
            T[3] += P["live"]
            if best:
                R["refuted"] += 1
                T[1] += 1
                R["first"] = R["first"] or [gname, P["first"]]
        print(f"{gname} {kind} {time.time() - t0:.0f}s", flush=True)
    for R in res.values():
        R["cosine"] = R["dot"] / max((R["nc"] * R["nx"]) ** 0.5, 1e-30)
        R["rel_diff"] = (R["nd"] / max(R["nc"], 1e-30)) ** 0.5
        for k in ("dot", "nc", "nx", "nd"):
            R.pop(k)
    out = {"gguf": f"{gguf_repo}/{fname}", "true": true_repo, "census": census, "cover": cover, "results": res,
               "imatrix_key": [k for k in h["kv"] if "imatrix" in k], "seconds": round(time.time() - t0)}
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
