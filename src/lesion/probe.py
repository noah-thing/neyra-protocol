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


def loop_spans(token_lists: list[list[int]], length: int = 1536, limit: int | None = None,
               minimum: int = 32) -> list[Span]:
    """Windows around where each text starts repeating, target being the repeated tokens."""
    spans = []
    for ids in token_lists:
        mask = repeated_mask(ids, 0)
        if mask.sum() < minimum:
            continue
        lo = max(0, int(np.argmax(mask)) - length // 2)
        m = mask[lo:lo + length]
        spans.append(Span(ids[lo:lo + length], 1, m, ~m))
    return spans[:limit]


class SelectivityProbe:
    """Mean drop in log probability on target tokens, minus the drop on kept tokens."""

    def __init__(self, runtime, spans: list[Span], keep_per_text: int = 96, seed: int = 0):
        self.rt, self.spans = runtime, spans
        rng = np.random.default_rng(seed)
        self.target, self.keep, self.positions = [], [], []
        for s in spans:
            target = np.flatnonzero(s.target[s.start:]) + s.start
            keep = np.flatnonzero(s.keep[s.start:]) + s.start
            if len(keep) > keep_per_text:
                keep = np.sort(rng.choice(keep, keep_per_text, replace=False))
            at = np.concatenate([target, keep])
            self.positions.append(at)
            self.target.append(np.arange(len(target)))
            self.keep.append(np.arange(len(target), len(at)))
        self.rt.reset()
        self.reference = self._read()

    def _read(self) -> list[np.ndarray]:
        return [self.rt.logprobs(s.tokens, s.start, p) for s, p in zip(self.spans, self.positions)]

    def drops(self) -> tuple[np.ndarray, np.ndarray]:
        now = self._read()
        target = [(r - n)[t].mean() if len(t) else 0.0 for r, n, t in zip(self.reference, now, self.target)]
        keep = [(r - n)[k].mean() if len(k) else 0.0 for r, n, k in zip(self.reference, now, self.keep)]
        return np.array(target), np.array(keep)

    def __call__(self) -> Measure:
        target, keep = self.drops()
        items = target - keep
        return Measure(float(items.mean()), float(items.std(ddof=1) / np.sqrt(len(items))), items)
