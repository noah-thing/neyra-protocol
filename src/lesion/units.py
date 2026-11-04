"""Split a patch into independently removable pieces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .patch import Patch
from .store import Checkpoint


@dataclass(frozen=True)
class Unit:
    tensor: str
    axis: str  # "tensor", "column" or "row"
    index: int

    def __str__(self) -> str:
        return self.tensor if self.axis == "tensor" else f"{self.tensor}[{self.axis} {self.index}]"


class UnitSet:
    def __init__(self, patch: Patch, base: Checkpoint, grain: str = "column"):
        self.base = {e.name: base.tensors[e.name].read() for e in patch.edits}
        self.edits = {e.name: e.dense(self.base[e.name]) for e in patch.edits}
        self.dtypes = {e.name: e.dtype for e in patch.edits}
        self.units: list[Unit] = []
        self.members: list[np.ndarray] = []
        for name, e in self.edits.items():
            if grain == "tensor" or len(e.shape) != 2:
                self.units.append(Unit(name, "tensor", 0))
                self.members.append(np.arange(len(e.idx)))
                continue
            key = e.idx % e.shape[1] if grain == "column" else e.idx // e.shape[1]
            order = np.argsort(key, kind="stable")
            values, starts = np.unique(key[order], return_index=True)
            for value, group in zip(values, np.split(order, starts[1:])):
                self.units.append(Unit(name, grain, int(value)))
                self.members.append(group)

    def __len__(self) -> int:
        return len(self.units)

    def weights_changed(self, chosen) -> int:
        return sum(len(self.members[i]) for i in chosen)

    def tensors(self, chosen) -> dict[str, tuple[np.ndarray, str]]:
        """Raw bits of every tensor with only the chosen units applied."""
        out = {}
        for i in chosen:
            name = self.units[i].tensor
            if name not in out:
                out[name] = self.base[name].copy().reshape(-1)
            e = self.edits[name]
            out[name][e.idx[self.members[i]]] = e.val[self.members[i]]
        return {n: (b.reshape(self.base[n].shape), self.dtypes[n]) for n, b in out.items()}
