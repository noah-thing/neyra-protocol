"""How much of a fine-tune's change is still in the weights after a hop.

The fine-tune moved a tensor by delta. After a merge, a conversion or a quantization the tensor
sits somewhere else, and its distance from the base is `change`. Survival is the projection of
change onto delta, over delta's squared norm: 1 when the change is all there, 0 when none of it
is, 2 when it was applied twice. It is a weights statement, so two implementations can agree on it.

Anything that rounds adds noise that is unrelated to delta. That noise does not move the estimate
on average, but it makes it uncertain, and `Pool.error` says by how much.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gguf_io import REPRODUCIBLE, SOURCES, encode, need_gguf


@dataclass
class Pool:
    """Survival over many tensors, weighted by how much each one was meant to move."""

    dot: float = 0.0
    norm: float = 0.0
    spread: float = 0.0  # sum over tensors of residual variance times the squared norm of delta
    moved: float = 0.0  # squared norm of everything that changed, wanted or not
    tensors: int = 0

    def add(self, change: np.ndarray, delta: np.ndarray) -> float:
        c, d = change.astype(np.float64).reshape(-1), delta.astype(np.float64).reshape(-1)
        norm = float(d @ d)
        if norm == 0.0:
            return float("nan")
        share = float(c @ d) / norm
        left = c - share * d
        self.dot += share * norm
        self.norm += norm
        self.moved += float(c @ c)
        self.spread += float(left @ left) / len(d) * norm
        self.tensors += 1
        return share

    @property
    def value(self) -> float | None:
        return self.dot / self.norm if self.norm else None

    @property
    def cosine(self) -> float | None:
        """How much the change that reached the file looks like the one that was trained, whatever its size."""
        return self.dot / math.sqrt(self.norm * self.moved) if self.norm and self.moved else None

    @property
    def error(self) -> float:
        """One standard error of the pooled estimate, from what is left over after removing the delta."""
        return math.sqrt(self.spread) / self.norm if self.norm else float("inf")


def quantized_copy(values: np.ndarray, kind: str, route: str = "direct") -> np.ndarray | None:
    """The values as a reproducible type stores them, read back. None for a type gguf-py cannot write."""
    if kind not in REPRODUCIBLE:
        return None
    gguf = need_gguf()
    if kind in ("F32", "F16"):
        return encode(SOURCES[route](values), kind).astype(np.float32)
    raw = encode(SOURCES[route](values), kind)
    return gguf.quants.dequantize(raw, getattr(gguf.GGMLQuantizationType, kind)).reshape(values.shape)


def predict(base: np.ndarray, delta: np.ndarray, kind: str) -> float | None:
    """What survives of delta if base plus delta is stored as `kind`: the change in the stored values,
    where base is stored the same way, projected on delta."""
    after = quantized_copy(base + delta.astype(base.dtype), kind)
    before = quantized_copy(base, kind)
    if after is None or before is None:
        return None
    pool = Pool()
    pool.add(after - before, delta)
    return pool.value


def forecast(base, fine, kinds: list[str], limit: int = 24) -> dict[str, Pool]:
    """Survival of an adapter's change in each type, before anything is converted.

    Reads the base and the adapter only. At most `limit` edited tensors are used, spread over the model,
    since a forecast does not need all of them.
    """
    from .numerics import to_float

    names = fine.names()
    names = [names[i] for i in sorted({round(i * (len(names) - 1) / max(limit - 1, 1)) for i in range(min(limit, len(names)))})]
    pools = {k: Pool() for k in kinds}
    for name in names:
        ref = base.tensors[name]
        w = to_float(ref.read(), ref.dtype).astype(np.float32)
        d = fine.delta(name).astype(np.float32)
        for kind in kinds:
            after, before = quantized_copy(w + d, kind), quantized_copy(w, kind)
            if after is not None and before is not None:
                pools[kind].add(after - before, d)
    return pools
