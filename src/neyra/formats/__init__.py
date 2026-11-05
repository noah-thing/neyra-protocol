"""Edit formats from other tools, read into and written out of Neyra patches.

Every importer ends the same way: compute the tensors the rival's own arithmetic produces, check any
digest the rival publishes, then let make_edit store the exact difference from the base. What the rival
did not publish a digest for is said in the patch meta, not assumed.
"""

from __future__ import annotations

import numpy as np

from ..patch import Patch, diff_tensor
from ..protocol import tensor_table
from ..store import Checkpoint


class Unsupported(ValueError):
    """The input is a real file of the format, but uses a feature this reader does not carry exactly."""


def build(base: Checkpoint, results: dict[str, np.ndarray], meta: dict) -> Patch:
    """A patch that turns base into base with the given tensors swapped in. Tensors that did not change are dropped."""
    edits = []
    for name, bits in sorted(results.items()):
        ref = base.tensors.get(name)
        if ref is None:
            raise Unsupported(f"{name}: not a tensor of the base")
        old = ref.read()
        bits = np.ascontiguousarray(bits).reshape(old.shape)
        if not np.array_equal(old, bits):
            edits.append(diff_tensor(name, ref.dtype, old, bits))
    return Patch(edits, meta).bind(tensor_table(base))


def new_bits(base: Checkpoint, patch: Patch) -> dict[str, np.ndarray]:
    return {e.name: e.apply(base.tensors[e.name].read()) for e in patch.edits}


NAMES = ("delta-league", "openwsp", "weightspace", "peft", "ballast")


def detect(path: str) -> str | None:
    """Guess the format of a file or folder from what is in it. Ballast deltas look like any checkpoint, so they are never guessed."""
    import json
    from pathlib import Path

    p = Path(path)
    if p.is_file() and p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            return "weightspace" if any(k.endswith("::idx") for k in z.files) else None
    if (p / "adapter_config.json").exists():
        return "peft"
    if (p / "manifest.json").exists():
        m = json.loads((p / "manifest.json").read_text())
        if m.get("schema_version") == "delta.submission.v1":
            return "delta-league"
        if m.get("format") == "openwsp.edit" or (p / "edit.npz").exists():
            return "openwsp"
    return None


def read(name: str, path: str, base: Checkpoint, **options) -> Patch:
    from . import ballast, deltaleague, openwsp, peft, weightspace

    readers = {"delta-league": deltaleague.load, "openwsp": openwsp.load, "weightspace": weightspace.load,
               "peft": peft.load, "ballast": ballast.load}
    if name not in readers:
        raise Unsupported(f"{name!r} is not one of {', '.join(NAMES)}")
    return readers[name](path, base, **options)
