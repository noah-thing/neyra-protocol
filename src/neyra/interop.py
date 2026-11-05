"""Read edits published in other formats."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .patch import Patch, PatchMismatch, TensorEdit
from .protocol import bits_digest, tensor_table
from .store import Checkpoint


def from_weight_space_logs(edit_dir: str | Path, base: Checkpoint) -> Patch:
    """Load an edit/ folder from the Sparks of Weight-Space Programming logs."""
    root = Path(edit_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(root / "edit.npz", allow_pickle=False)
    edits = []
    for i, t in enumerate(manifest["tensors"]):
        low_rank = t["form"] == "lowrank"
        bits = base.tensors[t["name"]].read()
        if hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest() != t["base_sha256"]:
            raise PatchMismatch(f"{t['name']}: not the base the published edit was made on")
        edit = TensorEdit(t["name"], "BF16", tuple(t["shape"]), bits_digest("BF16", bits), "",
                          z[f"t{i}_idx"].astype(np.int64), z[f"t{i}_val"],
                          z[f"t{i}_a"].astype(np.float64) if low_rank else None,
                          z[f"t{i}_b"].astype(np.float64) if low_rank else None)
        order = np.argsort(edit.idx, kind="stable")
        edit.idx, edit.val = edit.idx[order], edit.val[order]
        rebuilt = edit.apply(bits, verify=False)
        if hashlib.sha256(np.ascontiguousarray(rebuilt).tobytes()).hexdigest() != t["edited_sha256"]:
            raise PatchMismatch(f"{t['name']}: rebuilt tensor does not match the published hash", "result_mismatch")
        edit.result_digest = bits_digest("BF16", rebuilt)
        edits.append(edit)
    return Patch(edits, {"source": str(root)}).bind(tensor_table(base))
