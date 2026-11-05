#!/usr/bin/env python3
"""Is this checkpoint a fine-tune of the allowed base, or someone else's weights?

A judge for benchmarks that forbid starting from an instruction-tuned release (PostTrainBench rule 7). It reads a few
spans of tensors, over HTTP ranges for Hub repos, so a 7B model costs a few megabytes and no GPU. Only numpy.

    weights_identity_check.py --candidate final_model --base Qwen/Qwen3-1.7B-Base --instruct Qwen/Qwen3-1.7B

Verdicts, in the order they are tested:
  IDENTICAL_TO_INSTRUCT  every sampled span is bit for bit the instruct model's
  IDENTICAL_TO_BASE      every sampled span is bit for bit the base's (nothing was trained)
  BUILT_ON_INSTRUCT      the change from the base points where the instruct change points
  DERIVED_FROM_BASE      near the base, and not along the instruct change
  UNRELATED_TO_BASE      no more alike than two independently trained models
  INCONCLUSIVE           between the two ranges, or a model with another architecture
Exit status: 0 for DERIVED_FROM_BASE or IDENTICAL_TO_BASE, 1 for the rest, 2 for an error. A sample is not a proof of
equality: use --full to hash every tensor of a local candidate. The thresholds come from neyra's calibration
panel (see bench/identity_panel.py) and are printed with the result.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request

import numpy as np

SPAN = 16384
MATRICES = 24
# Calibrated on PolyPythias siblings and Qwen/SmolLM2 base-instruct fine-tunes; see bench/identity_panel.py.
UNRELATED_MAX = 0.10
DERIVED_MIN = 0.60
BUILT_ON_MIN = 0.50
ITEM = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
RAW = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


class Source:
    """The safetensors tensors of a local directory or a Hub repo, readable by span."""

    def __init__(self, spec):
        self.spec, self.files, self.tensors = spec, [], {}
        if os.path.isdir(spec):
            names = sorted(f for f in os.listdir(spec) if f.endswith(".safetensors") and not f.startswith("consolidated"))
            self.read = lambda f, o, n: _local(os.path.join(spec, f), o, n)
        else:
            sha = _json(f"https://huggingface.co/api/models/{spec}")["siblings"]
            names = sorted(s["rfilename"] for s in sha if s["rfilename"].endswith(".safetensors") and "/" not in s["rfilename"])
            self.read = lambda f, o, n: _remote(f"https://huggingface.co/{spec}/resolve/main/{f}", o, n)
        if not names:
            raise SystemExit(f"{spec}: no safetensors files")
        for f in names:
            size = struct.unpack("<Q", self.read(f, 0, 8))[0]
            header = json.loads(self.read(f, 8, size))
            for name, info in header.items():
                if name != "__metadata__" and info["dtype"] in ITEM:
                    self.tensors[name] = (f, info["dtype"], tuple(info["shape"]), 8 + size + info["data_offsets"][0],
                                          info["data_offsets"][1] - info["data_offsets"][0])

    def span(self, name, first, n):
        f, dtype, _, start, _ = self.tensors[name]
        raw = self.read(f, start + first * ITEM[dtype], n * ITEM[dtype])
        return _floats(np.frombuffer(raw, dtype=RAW[ITEM[dtype]]), dtype)

    def digest(self, name):
        f, _, _, start, size = self.tensors[name]
        h = hashlib.sha256()
        for off in range(0, size, 1 << 24):
            h.update(self.read(f, start + off, min(1 << 24, size - off)))
        return h.hexdigest()


def _local(path, offset, n):
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(n)


def _remote(url, offset, n):
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + n - 1}", "User-Agent": "weights-identity"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "weights-identity"}), timeout=60) as r:
        return json.load(r)


def _floats(bits, dtype):
    if dtype == "BF16":
        return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if dtype == "F16":
        return bits.view(np.float16).astype(np.float64)
    if dtype in ("F32", "F64"):
        return bits.view(np.float32 if dtype == "F32" else np.float64).astype(np.float64)
    return bits.astype(np.float64)


def _key(name):
    return hashlib.sha1(name.encode()).hexdigest()


def sample(*sources):
    """Matrices all sources hold with one shape, chosen by name hash alone."""
    first, rest = sources[0], sources[1:]
    ok = [n for n in sorted(first.tensors, key=_key) if len(first.tensors[n][2]) >= 2
          and all(n in s.tensors and s.tensors[n][2] == first.tensors[n][2] for s in rest)
          and first.tensors[n][1] in ("F32", "F16", "BF16")]
    return ok[:MATRICES]


def spans(names, *sources):
    for n in names:
        count = int(np.prod(sources[0].tensors[n][2]))
        k = min(SPAN, count)
        at = int(_key(n)[:8], 16) % (count - k + 1)
        yield n, [s.span(n, at, k) for s in sources]


def cos(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else None


def rel(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    na = np.linalg.norm(a)
    return float(np.linalg.norm(b - a) / na) if na else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def judge(cand, base, instruct=None):
    names = sample(base, cand, *([instruct] if instruct else []))
    if not names:
        return {"verdict": "INCONCLUSIVE", "why": "no matrix with the same name and shape in all models"}
    rows = [dict(zip("bcx" if instruct else "bc", v)) for _, v in spans(names, base, cand, *([instruct] if instruct else []))]
    out = {"spans": len(rows), "cos_to_base": med(cos(r["b"], r["c"]) for r in rows),
           "rel_to_base": med(rel(r["b"], r["c"]) for r in rows),
           "unchanged_from_base": float(np.mean([np.array_equal(r["b"], r["c"]) for r in rows]))}
    if instruct:
        out["unchanged_from_instruct"] = float(np.mean([np.array_equal(r["x"], r["c"]) for r in rows]))
        out["rel_to_instruct"] = med(rel(r["x"], r["c"]) for r in rows)
        out["rel_instruct_to_base"] = med(rel(r["b"], r["x"]) for r in rows)
        out["change_alignment"] = med(cos(r["c"] - r["b"], r["x"] - r["b"]) for r in rows)
        if out["unchanged_from_instruct"] == 1:
            return out | {"verdict": "IDENTICAL_TO_INSTRUCT"}
    if out["unchanged_from_base"] == 1:
        return out | {"verdict": "IDENTICAL_TO_BASE"}
    c = out["cos_to_base"]
    if c is None or UNRELATED_MAX < c < DERIVED_MIN:
        return out | {"verdict": "INCONCLUSIVE"}
    if c <= UNRELATED_MAX:
        return out | {"verdict": "UNRELATED_TO_BASE"}
    if instruct and (out["change_alignment"] or 0) >= BUILT_ON_MIN:
        return out | {"verdict": "BUILT_ON_INSTRUCT"}
    return out | {"verdict": "DERIVED_FROM_BASE"}


def same_everywhere(cand, other):
    shared = cand.tensors.keys() & other.tensors.keys()
    return not (cand.tensors.keys() ^ other.tensors.keys()) and all(cand.digest(n) == other.digest(n) for n in shared)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct")
    ap.add_argument("--full", action="store_true", help="hash every tensor to confirm an identical verdict")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        cand, base = Source(a.candidate), Source(a.base)
        inst = Source(a.instruct) if a.instruct else None
        r = judge(cand, base, inst)
        if a.full and r["verdict"].startswith("IDENTICAL"):
            ref = inst if r["verdict"] == "IDENTICAL_TO_INSTRUCT" else base
            r["every_tensor_hashed"] = same_everywhere(cand, ref)
            if not r["every_tensor_hashed"]:
                r["verdict"] = "DERIVED_FROM_BASE"
                r["why"] = "the sampled spans matched but a full hash did not"
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    r["sampled"] = f"{r.get('spans', 0)} spans of {SPAN} entries: a sample, not proof"
    r["thresholds"] = {"unrelated_max": UNRELATED_MAX, "derived_min": DERIVED_MIN, "built_on_min": BUILT_ON_MIN}
    print(json.dumps(r, indent=2) if a.json else "\n".join(f"{k}: {v}" for k, v in r.items()))
    return 0 if r["verdict"] in ("DERIVED_FROM_BASE", "IDENTICAL_TO_BASE") else 1


if __name__ == "__main__":
    sys.exit(main())
