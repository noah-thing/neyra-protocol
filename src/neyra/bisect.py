"""Find the part of a weight edit that causes a behavior.

Delta debugging (Zeller and Hildebrandt, 2002) with two changes for neural networks.
Behavior is a score with noise, not pass or fail, so a subset passes when it keeps a
chosen share of the full effect with confidence. And effects are often spread over
many units, so candidates are ranked first and the search then shrinks a prefix of
that ranking, which needs far fewer trials than splitting blindly.

The result is 1-minimal: removing any single remaining unit loses the effect. It is
not guaranteed to be the smallest set, and a different start can find another one.

The search looks at one set of items many times and keeps its best find, so its own
numbers are biased upward by selection. They are not evidence. Evidence comes from
confirm(), which measures the final set once on items the search never saw.

"Blame" can mean four different measurements: movement (q after a step minus q before it),
ablation (remove one part from the finished model), insertion (add one part alone to the base)
and replay (rerun the history without it). They answer different questions and can disagree.
Every trial here applies a subset of the edit to the base model and nothing else, so what
this module reports is insertion, and BLAME says so wherever a result is recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

Units = tuple[int, ...]
BLAME = "insertion"


@dataclass(frozen=True)
class Measure:
    value: float
    stderr: float = 0.0
    items: np.ndarray | None = field(default=None, compare=False)


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
    blame: str = BLAME

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
        if self.z == 0:
            return self.share(m) >= self.retain
        if m.items is not None and self.baseline.items is not None:
            y = retained_gap(self.baseline.items, m.items, self.full.items, self.retain)
            return y.mean() - self.z * y.std(ddof=1) / np.sqrt(len(y)) >= 0
        margin = self.z * np.hypot(m.stderr, self.baseline.stderr) / abs(self.effect)
        return self.share(m) - margin >= self.retain


def retained_gap(base: np.ndarray, subset: np.ndarray, full: np.ndarray, retain: float) -> np.ndarray:
    """Per item margin over the target. Its mean is at least 0 when the subset keeps enough."""
    sign = np.sign((full - base).mean()) or 1.0
    return sign * ((subset - base) - retain * (full - base))


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


def rank_banzhaf(search: Search, samples: int = 256, seed: int = 0) -> list[tuple[int, float]]:
    """Mean score with a unit minus mean score without it, over random subsets.

    Every sampled subset informs every unit, so cost does not grow with the number of
    units, and interactions count because units are scored among the others. See Wang
    and Jia (2023) on Banzhaf values with maximum sample reuse.
    """
    rng = np.random.default_rng(seed)
    n = search.n
    masks = rng.random((samples, n)) < 0.5
    values = np.array([search.share(search.ask(tuple(np.flatnonzero(m).tolist()))) for m in masks])
    inside = (masks * values[:, None]).sum(0) / np.maximum(masks.sum(0), 1)
    outside = ((~masks) * values[:, None]).sum(0) / np.maximum((~masks).sum(0), 1)
    return sorted(((u, float(inside[u] - outside[u])) for u in range(n)), key=lambda x: -x[1])


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


def bisect(n: int, measure: Callable[[Units], Measure], retain: float = 0.9, z: float = 0.0,
           budget: int = 2000, ranked: bool = True, banzhaf_above: int = 256) -> Result:
    search = Search(n, measure, retain, z, budget)
    ranking = [] if not ranked else rank_banzhaf(search) if n > banzhaf_above else rank(search)
    start = shrink_prefix(search, [u for u, _ in ranking]) if ranked else tuple(range(n))
    if not search.passes(start):
        start = tuple(range(n))
    units, minimal = ddmin(search, start)
    achieved = search.ask(units)
    return Result(tuple(sorted(units)), n, search.baseline, search.full, achieved,
                  search.share(achieved), minimal, search.trials, ranking)


@dataclass
class Confirmation:
    n: int
    retained: float
    margin: float
    low: float
    passed: bool


def confirm(n: int, units: Units, measure: Callable[[Units], Measure], retain: float = 0.9,
            boot: int = 10000, seed: int = 0) -> Confirmation:
    """One look, on items the search never saw. Passes if the lower bound clears the target."""
    base, subset, full = measure(()), measure(tuple(units)), measure(tuple(range(n)))
    y = retained_gap(base.items, subset.items, full.items, retain)
    rng = np.random.default_rng(seed)
    means = y[rng.integers(0, len(y), (boot, len(y)))].mean(axis=1)
    low = float(np.quantile(means, 0.05))
    kept = float((subset.items - base.items).mean() / (full.items - base.items).mean())
    return Confirmation(len(y), kept, float(y.mean()), low, low >= 0)
