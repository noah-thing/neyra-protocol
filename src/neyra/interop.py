"""Read edits published in other formats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .patch import Patch, TensorEdit


def from_weight_space_logs(edit_dir: str | Path) -> Patch:
    """Load an edit/ folder from the Sparks of Weight-Space Programming logs."""
    root = Path(edit_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    z = np.load(root / "edit.npz", allow_pickle=False)
    edits = []
    for i, t in enumerate(manifest["tensors"]):
        low_rank = t["form"] == "lowrank"
        edits.append(TensorEdit(
            t["name"], "BF16", tuple(t["shape"]), t["base_sha256"], t["edited_sha256"],
            z[f"t{i}_idx"].astype(np.int64), z[f"t{i}_val"],
            z[f"t{i}_a"].astype(np.float64) if low_rank else None,
            z[f"t{i}_b"].astype(np.float64) if low_rank else None,
        ))
    return Patch(edits, {"source": str(root)})
