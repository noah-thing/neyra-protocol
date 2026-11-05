"""wvc manifests. wvc's tensor id is the SHA-256 of the stored bytes, and its root hashes the sorted lines.

Writing the manifest of a patch's base and result lets a wvc user check that a Neyra patch starts and ends
on the checkpoints they hold. wvc's own store of zstd deltas is not read.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..patch import Patch
from ..store import Checkpoint

FORMAT = "wvc-manifest/0"
ESCAPES = {"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r"}


def line(name: str, dtype: str, shape: tuple[int, ...], sha: str) -> str:
    escaped = "".join(ESCAPES.get(c, c) for c in name)
    return f"{escaped}\t{dtype}\t{'x'.join(map(str, shape))}\t{sha}\n"


def build(entries: dict[str, dict]) -> dict:
    root = hashlib.sha256()
    for name in sorted(entries, key=str.encode):
        e = entries[name]
        root.update(line(name, e["dtype"], tuple(e["shape"]), e["sha256"]).encode())
    return {"format": FORMAT, "root": root.hexdigest(), "tensor_count": len(entries),
            "total_bytes": sum(e["nbytes"] for e in entries.values()),
            "tensors": {n: entries[n] for n in sorted(entries, key=str.encode)}}


def manifest(ck: Checkpoint, replaced: dict[str, np.ndarray] | None = None) -> dict:
    """The manifest of a checkpoint, with some tensors swapped for the given bits."""
    entries = {}
    for name, ref in ck.tensors.items():
        sha = (hashlib.sha256(np.ascontiguousarray(replaced[name]).tobytes()).hexdigest()
               if replaced and name in replaced else ref.sha256())
        entries[name] = {"dtype": ref.dtype, "shape": list(ref.shape), "nbytes": ref.nbytes, "sha256": sha}
    return build(entries)


def dump(patch: Patch, base: Checkpoint) -> tuple[dict, dict]:
    """Manifests of the base and of the result of the patch."""
    result = {e.name: e.apply(base.tensors[e.name].read()) for e in patch.edits}
    return manifest(base), manifest(base, result)
