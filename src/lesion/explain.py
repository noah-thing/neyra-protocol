"""Read what a residual direction means by passing it through the model's output head.

This is the logit lens (nostalgebraist, 2020) applied to a direction instead of a
hidden state. It is a first order reading of weights alone: it says which tokens a
direction favours at the output, not what it does through later layers.
"""

from __future__ import annotations

import re

import numpy as np

from .numerics import to_float
from .store import Checkpoint


def _find(ck: Checkpoint, pattern: str):
    hits = [n for n in ck.tensors if re.search(pattern, n) and "visual" not in n and "mtp" not in n]
    return ck.tensors[hits[0]] if hits else None


def output_matrix(ck: Checkpoint) -> np.ndarray:
    ref = _find(ck, r"(^|\.)lm_head\.weight$") or _find(ck, r"embed_tokens\.weight$")
    if ref is None:
        raise KeyError("no output head or embedding in this checkpoint")
    return to_float(ref.read(), ref.dtype)


def final_gain(ck: Checkpoint) -> np.ndarray | None:
    ref = _find(ck, r"(^|\.)model(\.language_model)?\.norm\.weight$")
    if ref is None:
        return None
    g = to_float(ref.read(), ref.dtype).astype(np.float64)
    # Some families store the RMSNorm gain as an offset from one.
    return 1 + g if abs(g.mean()) < 0.5 else g


def readout(ck: Checkpoint, direction: np.ndarray, k: int = 12, matrix: np.ndarray | None = None) -> tuple[list, list]:
    """Token ids most favoured and most disfavoured by a direction, with scores in standard deviations."""
    e = output_matrix(ck) if matrix is None else matrix
    gain = final_gain(ck)
    v = direction if gain is None else direction * gain
    logits = e.astype(np.float32) @ v.astype(np.float32)
    z = (logits - logits.mean()) / logits.std()
    order = np.argsort(z)
    return ([(int(i), float(z[i])) for i in order[::-1][:k]], [(int(i), float(z[i])) for i in order[:k]])
