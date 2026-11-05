"""The weightspace package's sparse patch: an .npz of "<tensor>::idx" and "<tensor>::val" arrays.

The file carries no digests of the model it was made on, so the patch binds to whatever base the caller gives.
Values for a bfloat16 tensor are stored widened to float32, which is exact, and are rounded back on import.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..numerics import from_float, is_float, to_float
from ..patch import Patch
from ..store import Checkpoint
from . import Unsupported, build

IDX, VAL = "::idx", "::val"


def load(path: str | Path, base: Checkpoint) -> Patch:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    names = sorted(k[: -len(IDX)] for k in arrays if k.endswith(IDX))
    if not names or any(n + VAL not in arrays for n in names) or len(arrays) != 2 * len(names):
        raise Unsupported(f"{path}: expected <name>::idx and <name>::val pairs and nothing else")
    results = {}
    for name in names:
        ref = base.tensors.get(name)
        if ref is None or not is_float(ref.dtype):
            raise Unsupported(f"{name}: not a float tensor of the base")
        idx, val = arrays[name + IDX], arrays[name + VAL]
        if idx.dtype.kind not in "iu" or idx.ndim != 1 or val.shape != idx.shape or val.dtype.kind != "f":
            raise Unsupported(f"{name}: bad index or value arrays")
        if idx.size and (idx.min() < 0 or idx.max() >= ref.numel or len(np.unique(idx)) != idx.size):
            raise Unsupported(f"{name}: indices outside the tensor, or repeated")
        old = ref.read()
        bits = old.reshape(-1).copy()
        bits[idx.astype(np.int64)] = from_float(val.astype(np.float32), ref.dtype)
        results[name] = bits.reshape(old.shape)
    return build(base, results, {"source": str(path), "format": "weightspace-npz",
                                 "digests": "none published, bound to the base given"})


def dump(patch: Patch, base: Checkpoint, out: str | Path) -> None:
    arrays = {}
    for e in patch.edits:
        full = e.dense(base.tensors[e.name].read())
        arrays[e.name + IDX] = full.idx
        if e.dtype in ("BF16", "F16", "F32"):
            arrays[e.name + VAL] = to_float(full.val, e.dtype).astype(np.float16 if e.dtype == "F16" else np.float32)
        else:
            raise Unsupported(f"{e.name}: {e.dtype} is not a float dtype")
    np.savez_compressed(out, **arrays)
