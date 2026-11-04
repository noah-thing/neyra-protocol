"""Fast deterministic scores from teacher forcing a fixed set of texts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bisect import Measure


def repeated_mask(tokens: list[int], start: int, n: int = 12) -> np.ndarray:
    """True at each position whose preceding n tokens, itself included, appeared earlier."""
    out = np.zeros(len(tokens), dtype=bool)
    seen: set[tuple[int, ...]] = set()
    for i in range(start + n, len(tokens)):
        gram = tuple(tokens[i - n + 1:i + 1])
        out[i] = gram in seen
        seen.add(gram)
    return out


@dataclass
class Span:
    tokens: list[int]
    start: int
    target: np.ndarray  # positions the edit should make less likely
    keep: np.ndarray  # positions it should leave alone


class SelectivityProbe:
    """Mean drop in log probability on target tokens, minus the drop on kept tokens."""

    def __init__(self, runtime, spans: list[Span]):
        self.rt, self.spans = runtime, spans
        self.rt.reset()
        self.reference = [self.rt.logprobs(s.tokens, s.start) for s in spans]

    def drops(self) -> tuple[np.ndarray, np.ndarray]:
        target, keep = [], []
        for s, ref in zip(self.spans, self.reference):
            d = ref - self.rt.logprobs(s.tokens, s.start)
            target.append(d[s.target[s.start:]].mean() if s.target[s.start:].any() else 0.0)
            keep.append(d[s.keep[s.start:]].mean() if s.keep[s.start:].any() else 0.0)
        return np.array(target), np.array(keep)

    def __call__(self) -> Measure:
        target, keep = self.drops()
        per_text = target - keep
        return Measure(float(per_text.mean()), 0.0)
