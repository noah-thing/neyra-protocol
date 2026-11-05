"""Read edits published in other formats. The readers live in neyra.formats, one module per format."""

from __future__ import annotations

from pathlib import Path

from .formats import openwsp
from .patch import Patch
from .store import Checkpoint


def from_weight_space_logs(edit_dir: str | Path, base: Checkpoint) -> Patch:
    """Load an edit/ folder from the Sparks of Weight-Space Programming logs."""
    return openwsp.load(edit_dir, base)
