"""Find the part of a weight edit that causes a behavior.

Delta debugging (Zeller and Hildebrandt, 2002) with two changes for neural networks.
Behavior is a score with noise, not pass or fail, so a subset passes when it keeps a
chosen share of the full effect with confidence. And effects are often spread over
many units, so candidates are ranked first and the search then shrinks a prefix of
that ranking, which needs far fewer trials than splitting blindly.

The result is 1-minimal: removing any single remaining unit loses the effect. It is
not guaranteed to be the smallest set, and a different start can find another one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

Units = tuple[int, ...]


@dataclass(frozen=True)
class Measure:
    value: float
    stderr: float = 0.0


@dataclass
class Trial:
    units: Units
    measure: Measure
    retained: float


@dataclass
class Result:
    units: Units
    total: int
    baseline: Measure
    full: Measure
    achieved: Measure
    retained: float
    minimal: bool
    trials: list[Trial] = field(default_factory=list)
    ranking: list[tuple[int, float]] = field(default_factory=list)

    @property
    def effect(self) -> float:
        return self.full.value - self.baseline.value


class NoEffect(ValueError):
    pass


class Search:
    def __init__(self, n: int, measure: Callable[[Units], Measure], retain: float, z: float, budget: int):
        self.n, self.measure, self.retain, self.z, self.budget = n, measure, retain, z, budget
        self.cache: dict[Units, Measure] = {}
        self.trials: list[Trial] = []
        self.baseline = self.ask(())
        self.full = self.ask(tuple(range(n)))
        self.effect = self.full.value - self.baseline.value
        floor = z * np.hypot(self.baseline.stderr, self.full.stderr)
        if abs(self.effect) <= floor:
            raise NoEffect(f"full edit moved the score by {self.effect:.4g}, within noise of {floor:.4g}")

    def ask(self, units: Units) -> Measure:
        units = tuple(sorted(units))
        if units not in self.cache:
            self.cache[units] = self.measure(units)
            if hasattr(self, "effect"):
                self.trials.append(Trial(units, self.cache[units], self.share(self.cache[units])))
        return self.cache[units]

    def share(self, m: Measure) -> float:
        return (m.value - self.baseline.value) / self.effect

    @property
    def spent(self) -> bool:
        return len(self.cache) >= self.budget

    def passes(self, units: Units) -> bool:
        m = self.ask(units)
        margin = self.z * np.hypot(m.stderr, self.baseline.stderr) / abs(self.effect)
        return self.share(m) - margin >= self.retain


def rank(search: Search) -> list[tuple[int, float]]:
    """Each unit's share of the effect: alone, and how much is lost without it."""
    everything = tuple(range(search.n))
    scores = []
    for u in everything:
        if search.spent:
            scores.append((u, 0.0))
            continue
        alone = search.share(search.ask((u,)))
        lost = 1.0 - search.share(search.ask(tuple(x for x in everything if x != u)))
        scores.append((u, (alone + lost) / 2))
    return sorted(scores, key=lambda x: -x[1])


def shrink_prefix(search: Search, order: Sequence[int]) -> Units:
    """Smallest prefix of the ranking that keeps the effect, by binary search."""
    lo, hi = 1, len(order)
    while lo < hi and not search.spent:
        mid = (lo + hi) // 2
        if search.passes(tuple(order[:mid])):
            hi = mid
        else:
            lo = mid + 1
    return tuple(order[:hi])


def ddmin(search: Search, start: Units) -> tuple[Units, bool]:
    """Classic reduction to a 1-minimal set. Returns the set and whether that was proven."""
    current = list(start)
    chunks = 2
    while len(current) >= 2:
        if search.spent:
            return tuple(current), False
        parts = np.array_split(np.array(current), min(chunks, len(current)))
        reduced = False
        for part in parts:
            if len(part) < len(current) and search.passes(tuple(int(x) for x in part)):
                current, chunks, reduced = [int(x) for x in part], 2, True
                break
        if not reduced:
            for part in parts:
                rest = [x for x in current if x not in set(part.tolist())]
                if rest and search.passes(tuple(rest)):
                    current, chunks, reduced = rest, max(chunks - 1, 2), True
                    break
        if not reduced:
            if chunks >= len(current):
                return tuple(current), True
            chunks = min(chunks * 2, len(current))
    return tuple(current), True


def bisect(n: int, measure: Callable[[Units], Measure], retain: float = 0.9, z: float = 1.96,
           budget: int = 2000, ranked: bool = True) -> Result:
    search = Search(n, measure, retain, z, budget)
    ranking = rank(search) if ranked else []
    start = shrink_prefix(search, [u for u, _ in ranking]) if ranked else tuple(range(n))
    if not search.passes(start):
        start = tuple(range(n))
    units, minimal = ddmin(search, start)
    achieved = search.ask(units)
    return Result(tuple(sorted(units)), n, search.baseline, search.full, achieved,
                  search.share(achieved), minimal, search.trials, ranking)
