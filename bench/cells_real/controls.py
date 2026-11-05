"""Controls: llama.cpp's own quantizer (ggml-quants.c compiled here) on every real tensor of Qwen3-0.6B.

usage: python -m bench.cells_real.controls TYPE [requant]
TYPE is Q4_K, Q5_K, Q6_K or Q8_0. For each 2-D tensor wide enough, quantize the true bfloat16 source with and without a
random importance matrix and count source values outside the stored cells. With `requant`, first round the source
through Q8_0 and dequantize, quantize that to TYPE, and count outside cells for both the Q8_0 values (must be 0) and the
original source (the case a requantized file presents).
"""
import ctypes
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from neyra import cells

from .lib import Safetensors, bf16_to_f32

KEEP = ("ggml/src/ggml-quants.c", "ggml/src/ggml-quants.h", "ggml/src/ggml-common.h", "ggml/src/ggml-impl.h", "ggml/src/ggml-cpu/ggml-cpu-impl.h")
SIZES = {"Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_0": (32, 34)}
STUB = r'''
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"
void ggml_abort(const char*f,int l,const char*m,...){abort();}
size_t ggml_row_size(enum ggml_type t,int64_t n){switch(t){case GGML_TYPE_Q8_0:return n/32*34;case GGML_TYPE_Q4_K:return n/256*144;
case GGML_TYPE_Q5_K:return n/256*176;case GGML_TYPE_Q6_K:return n/256*210;default:abort();}}
const char*ggml_type_name(enum ggml_type t){return "";}
size_t ggml_type_size(enum ggml_type t){return 0;}
'''


def build():
    work = tempfile.mkdtemp(prefix="llamaq_")
    data = urllib.request.urlopen("https://github.com/ggml-org/llama.cpp/archive/10d4846a572aebf21c0482d140b00e74c0f1ad20.tar.gz", timeout=300).read()
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if rel in KEEP or rel.startswith("ggml/include/"):
                m.name = rel
                tf.extract(m, work)
    del data
    Path(f"{work}/stubs.c").write_text(STUB)
    lib = f"{work}/q.dylib"
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", f"-I{work}/ggml/include", f"-I{work}/ggml/src", f"{work}/ggml/src/ggml-quants.c",
                    f"{work}/stubs.c", "-o", lib], check=True)
    return ctypes.CDLL(lib), work


def quant(lib, kind, x, im):
    f = getattr(lib, f"quantize_{kind.lower().replace('k', 'K')}")
    f.restype = ctypes.c_size_t
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    w, b = SIZES[kind]
    x = np.ascontiguousarray(x, np.float32)
    out = np.zeros(x.shape[0] * x.shape[1] // w * b, np.uint8)
    f(x.ctypes.data, out.ctypes.data, x.shape[0], x.shape[1], None if im is None else im.ctypes.data)
    return out.reshape(x.shape[0], -1)


def dequant_q8(lib, raw, shape):
    f = lib.dequantize_row_q8_0
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    out = np.zeros(shape, np.float32)
    raw = np.ascontiguousarray(raw)
    f(raw.ctypes.data, out.ctypes.data, out.size)
    return out


def main():
    kind = sys.argv[1]
    requant = "requant" in sys.argv
    t0 = time.time()
    lib, work = build()
    st = Safetensors("Qwen/Qwen3-0.6B")
    rng = np.random.default_rng(5)
    modes = ["plain", "imatrix"] if not requant else ["requant"]
    tot = {m: {"tensors": 0, "elems": 0, "tensors_refuted": 0, "outside": 0, "worst_tensor": None} for m in modes}
    if requant:
        tot["requant"].update(intermediate_outside=0)
    for name in sorted(st.meta):
        meta = st.meta[name]
        if len(meta["shape"]) != 2 or meta["shape"][1] % SIZES[kind][0]:
            continue
        w = bf16_to_f32(st.read(name)[0])
        step = max(1, 2_000_000 // w.shape[1])
        res = {m: [0, 0] for m in modes}
        for r0 in range(0, w.shape[0], step):
            x = np.ascontiguousarray(w[r0:r0 + step])
            for m in modes:
                if m == "requant":
                    q8 = quant(lib, "Q8_0", x, None)
                    mid = dequant_q8(lib, q8, x.shape)
                    raw = quant(lib, kind, mid, None)
                    c = cells.decode(kind, raw)
                    tot[m]["intermediate_outside"] += len(cells.violations(c, mid))
                else:
                    im = (rng.random(x.shape[1]).astype(np.float32) + 0.05) if m == "imatrix" and kind != "Q8_0" else None
                    c = cells.decode(kind, quant(lib, kind, x, im))
                res[m][0] += len(cells.violations(c, x.reshape(-1)))
                res[m][1] += x.size
        for m in modes:
            T = tot[m]
            T["tensors"] += 1
            T["elems"] += res[m][1]
            T["outside"] += res[m][0]
            if res[m][0]:
                T["tensors_refuted"] += 1
                T["worst_tensor"] = T["worst_tensor"] or name
        print(name, {m: res[m][0] for m in modes}, f"{time.time() - t0:.0f}s", flush=True)
    print("RESULT " + json.dumps({"kind": kind, "requant": requant, "totals": tot, "llama_cpp": "10d4846a572a", "seconds": round(time.time() - t0)}), flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
