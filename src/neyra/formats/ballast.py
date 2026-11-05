"""A ballast delta: a directory of safetensors holding model minus base, which `ballast apply` adds back.

ballast subtracts and adds in float32 and rounds to the dtype of the tensor it writes. That is the rule here.
Subtracting in a 16 bit dtype can lose bits, so a delta made from two models need not rebuild the second
exactly. A patch made here records what `ballast apply` gives, and dump says which tensors would not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..numerics import from_float, is_float, to_float
from ..patch import Patch, write_safetensors
from ..store import Checkpoint
from . import Unsupported, build


def load(path: str | Path, base: Checkpoint) -> Patch:
    delta = Checkpoint.open(str(path))
    results = {}
    for name, ref in delta.tensors.items():
        target = base.tensors.get(name)
        if target is None:
            raise Unsupported(f"{name}: not a tensor of the base")
        if target.shape != ref.shape or not is_float(target.dtype) or not is_float(ref.dtype):
            raise Unsupported(f"{name}: delta and base must be float tensors of one shape")
        total = to_float(target.read(), target.dtype).astype(np.float32) + to_float(ref.read(), ref.dtype).astype(np.float32)
        results[name] = from_float(total, target.dtype)
    return build(base, results, {"source": str(path), "format": "ballast-delta",
                                 "digests": "none published, bound to the base given"})


def dump(patch: Patch, base: Checkpoint, out: str | Path) -> list[str]:
    """Write model minus base. Returns the tensors `ballast apply` would not rebuild exactly."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tensors, inexact, kinds = {}, [], set()
    for e in patch.edits:
        if e.dtype not in ("BF16", "F16", "F32"):
            raise Unsupported(f"{e.name}: {e.dtype} is not a float dtype")
        old = base.tensors[e.name].read()
        new = e.apply(old)
        d = from_float(to_float(new, e.dtype).astype(np.float32) - to_float(old, e.dtype).astype(np.float32), e.dtype)
        back = from_float(to_float(old, e.dtype).astype(np.float32) + to_float(d, e.dtype).astype(np.float32), e.dtype)
        if not np.array_equal(back, new):
            inexact.append(e.name)
        tensors[e.name] = d.view(np.float32) if e.dtype == "F32" else d
        if e.dtype != "F32":
            kinds.add(e.dtype)
    if len(kinds) > 1:
        raise Unsupported("BF16 and F16 tensors cannot share one file here")
    write_safetensors(out / "model.safetensors", tensors, {"format": "pt"}, kinds.pop() if kinds else None)
    return inexact
