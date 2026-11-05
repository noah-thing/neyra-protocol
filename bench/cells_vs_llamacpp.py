"""Check neyra.cells against llama.cpp's own ggml-quants.c, compiled from source.

usage: cells_vs_llamacpp.py /path/to/llama.cpp/ggml [fp-contract: off|on] [--emit tests/fixtures/ggml_cells.npz]

Per type, on random float32 data (heavy tails, zero blocks, and with and without an importance matrix):
  - the dequantized values neyra.cells computes equal ggml's dequantize_row_* bit for bit;
  - the source value lies inside the cell for every element, with the slack in neyra.cells.SLACK.
"""
import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from neyra import cells

TYPES = {"Q8_0": 34, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210}


def build(root: Path, contract: str) -> ctypes.CDLL:
    out = Path(tempfile.mkdtemp()) / "q.dylib"
    stubs = Path(tempfile.mkdtemp()) / "stubs.c"
    stubs.write_text('#include <stdio.h>\n#include <stdlib.h>\n#include "ggml.h"\n'
                     'void ggml_abort(const char*f,int l,const char*m,...){abort();}\n'
                     'size_t ggml_row_size(enum ggml_type t,int64_t n){switch(t){'
                     'case GGML_TYPE_Q4_0:return n/32*18;case GGML_TYPE_Q4_1:return n/32*20;'
                     'case GGML_TYPE_Q5_0:return n/32*22;case GGML_TYPE_Q5_1:return n/32*24;'
                     'case GGML_TYPE_Q8_0:return n/32*34;case GGML_TYPE_Q4_K:return n/256*144;'
                     'case GGML_TYPE_Q5_K:return n/256*176;case GGML_TYPE_Q6_K:return n/256*210;'
                     'default:abort();}}\n'
                     'const char*ggml_type_name(enum ggml_type t){return "";}\n'
                     'size_t ggml_type_size(enum ggml_type t){return 0;}\n')
    src = root / "src"
    subprocess.run(["cc", "-O2", f"-ffp-contract={contract}", "-shared", "-fPIC", f"-I{root / 'include'}", f"-I{src}",
                    str(src / "ggml-quants.c"), str(stubs), "-o", str(out)], check=True)
    return ctypes.CDLL(str(out))


def sample(rng, n: int, kind: str, scale: float = 0.02) -> np.ndarray:
    x = rng.standard_normal(n).astype(np.float32) * scale
    x[rng.random(n) < 0.01] *= 20
    if kind != "Q4_0":
        x[: 256 * 3] = 0
    # a block with a single large entry and a block with one-signed values
    x[256 * 5] = 3.0
    x[256 * 6: 256 * 7] = np.abs(x[256 * 6: 256 * 7])
    return x


def main():
    root = Path(sys.argv[1])
    contract = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "off"
    emit = sys.argv[sys.argv.index("--emit") + 1] if "--emit" in sys.argv else None
    fixture = {}
    lib = build(root, contract)
    rng = np.random.default_rng(7)
    n = 256 * 400
    print(f"fp-contract={contract}")
    print(f"{'type':6} {'mode':8} {'decode mismatches':>18} {'outside cell (slack)':>21} {'outside, no slack':>18}  worst overshoot / step")
    for kind, width, use_im, scale in [(k, w, i, sc) for k, w in TYPES.items() for i in (False, True) for sc in (0.02, 0.0005)]:
        if True:
            if use_im and kind in ("Q8_0",):
                continue
            mode = ("im" if use_im else "plain") if scale > 0.001 else "tiny"
            if use_im and scale < 0.001:
                continue
            x = sample(rng, n, kind, scale)
            im = (rng.random(256).astype(np.float32) + 0.1) if use_im else None
            nrows, per = n // 256, 256
            dst = np.zeros(n // (256 if kind.endswith("K") else 32) * width, np.uint8)
            fn = getattr(lib, f"quantize_{kind.lower().replace('k', 'K')}")
            fn.restype = ctypes.c_size_t
            fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
            fn(x.ctypes.data, dst.ctypes.data, nrows, per, im.ctypes.data if im is not None else None)
            ref = np.zeros(n, np.float32)
            dq = getattr(lib, f"dequantize_row_{kind.lower().replace('k', 'K')}")
            dq.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            dq(dst.ctypes.data, ref.ctypes.data, n)
            if emit:
                fixture[f"{kind}_{mode}_x"] = x[: 8 * 256]
                fixture[f"{kind}_{mode}_raw"] = dst[: (8 * 256 // (256 if kind.endswith('K') else 32)) * width]
                fixture[f"{kind}_{mode}_ref"] = ref[: 8 * 256]
            c = cells.decode(kind, dst)
            mism = int((c.centre.view(np.uint32) != ref.view(np.uint32)).sum())
            out = len(cells.violations(c, x))
            step = np.where(np.isfinite(c.low) & np.isfinite(c.high), (c.high - c.low) / (1 + 2 * cells.SLACK), 1)
            over = np.maximum(np.maximum(c.low - x, x - c.high), 0)[c.live] / np.maximum(step[c.live], 1e-30)
            bare = c.centre.astype(np.float64)
            tight = ((x < bare - step / 2) & np.isfinite(c.low)) | ((x > bare + step / 2) & np.isfinite(c.high))
            print(f"{kind:6} {mode:8} {mism:18d} {out:21d} {int((tight & c.live).sum()):18d}  {over.max():.2e}")
    if emit:
        np.savez_compressed(emit, **fixture)


if __name__ == "__main__":
    main()
