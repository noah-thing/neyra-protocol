"""A PEFT or mlx-lm LoRA adapter, as the patch its merge writes.

The adapter alone does not say what the merged bits are, because each tool rounds differently. The caller
names the merge path, default PEFT's merge_and_unload(), and the patch records it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..check import RECIPES, Adapter, targets
from ..patch import Patch
from ..store import Checkpoint
from . import Unsupported, build


def load(path: str | Path, base: Checkpoint, merge: str = "peft") -> Patch:
    if merge not in RECIPES:
        raise Unsupported(f"merge path {merge!r} is not one of {sorted(RECIPES)}")
    adapter = Adapter(Checkpoint.open(str(path)))
    results: dict[str, np.ndarray] = {}
    for name, key in targets(adapter, base).items():
        ref = base.tensors[name]
        term = adapter.term(key)
        if term.a.shape[1] != ref.shape[-1] or term.b.shape[0] != ref.shape[0]:
            raise Unsupported(f"{name}: the adapter's shapes do not fit the base tensor")
        results[name] = RECIPES[merge](ref.dtype, ref.read(), term)
    return build(base, results, {"source": str(path), "format": "peft-lora", "merge": merge,
                                 "digests": "none published, the merged bits follow the named merge path"})
