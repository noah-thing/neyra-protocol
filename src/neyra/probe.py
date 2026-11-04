"""Fast deterministic scores from teacher forcing a fixed set of texts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .behavior import loop_onset, repeat_flags
from .bisect import Measure


@dataclass
class Span:
    tokens: list[int]
    start: int
    target: np.ndarray  # positions the edit should make less likely
    keep: np.ndarray  # positions it should leave alone


def loop_spans(token_lists: list[list[int]], length: int = 1024, limit: int | None = None,
               lead: int = 200) -> list[Span]:
    """Windows opening just before each terminal loop. Target is the repeated tokens inside.

    Anchoring on the first repeated phrase is wrong: careful reasoning restates itself
    long before anything loops, so such a window holds no loop at all.
    """
    spans = []
    for ids in token_lists:
        onset = loop_onset(ids)
        if onset is None:
            continue
        lo = max(0, onset - lead)
        window = ids[lo:lo + length]
        target = np.array(repeat_flags(ids)[lo:lo + length])
        target[:max(onset - lo, 0)] = False
        spans.append(Span(window, 1, target, ~target))
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
