"""Find a behavior's direction from examples and remove it from the columns that write it.

Nothing here is new as a technique. The direction is a difference of means (Arditi et
al., 2024, for refusal), and the edit is the rank 1 column update of Steer2Edit (Sun et
al., 2026) and of Experiment 8 in Sparks of Weight-Space Programming. The point is that
every step is a measurement a script can run, so the search needs no judgement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .numerics import from_float, to_float
from .patch import Patch, diff_tensor
from .probe import SelectivityProbe, Span
from .store import Checkpoint


def down_proj_name(ck: Checkpoint, layer: int) -> str:
    suffix = f"layers.{layer}.mlp.down_proj.weight"
    hits = [n for n in ck.tensors if n.endswith(suffix) and "visual" not in n and "mtp" not in n]
    if len(hits) != 1:
        raise KeyError(f"expected one tensor ending in {suffix}, found {hits}")
    return hits[0]


def project_columns(bits: np.ndarray, dtype: str, u: np.ndarray, columns: np.ndarray, alpha: float) -> np.ndarray:
    """W' = W - alpha u (u^T W) restricted to the given columns."""
    w = to_float(bits, dtype).astype(np.float64)
    block = w[:, columns]
    w[:, columns] = block - alpha * np.outer(u, u @ block)
    return from_float(w.astype(np.float32), dtype)


@dataclass
class Layer:
    index: int
    direction: np.ndarray
    columns: np.ndarray
    write: np.ndarray  # signed contribution of every column to the direction


@dataclass
class Plan:
    layers: list[Layer]
    alpha: float
    score: float
    trace: list[dict] = field(default_factory=list)


def collect(runtime, spans: list[Span], layers: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Mean residual and MLP activity over target and kept tokens, per layer."""
    sums = {i: {"t": 0.0, "k": 0.0, "at": 0.0, "ak": 0.0, "nt": 0, "nk": 0} for i in layers}
    for s in spans:
        resid = runtime.capture(s.tokens, layers, "resid")
        acts = runtime.capture(s.tokens, layers, "mlp")
        for i in layers:
            a = sums[i]
            a["t"] = a["t"] + resid[i][s.target].sum(axis=0)
            a["k"] = a["k"] + resid[i][s.keep].sum(axis=0)
            a["at"] = a["at"] + acts[i][s.target].sum(axis=0)
            a["ak"] = a["ak"] + acts[i][s.keep].sum(axis=0)
            a["nt"] += int(s.target.sum())
            a["nk"] += int(s.keep.sum())
    return {i: {"target": a["t"] / a["nt"], "keep": a["k"] / a["nk"],
                "act_target": a["at"] / a["nt"], "act_keep": a["ak"] / a["nk"]}
            for i, a in sums.items()}


def writers(bits: np.ndarray, dtype: str, stats: dict[str, np.ndarray], k: int) -> tuple:
    diff = stats["target"] - stats["keep"]
    u = diff / np.linalg.norm(diff)
    w = to_float(bits, dtype).astype(np.float64)
    write = (u @ w) * (stats["act_target"] - stats["act_keep"])
    return u, np.sort(np.argsort(-write)[:k]), write


class Repair:
    def __init__(self, runtime, ck: Checkpoint, spans: list[Span], probe_spans: list[Span], columns: int = 64):
        self.rt, self.ck, self.k = runtime, ck, columns
        self.spans = spans
        self.probe = SelectivityProbe(runtime, probe_spans)
        self.base = {}

    def _bits(self, layer: int) -> tuple[str, np.ndarray, str]:
        name = down_proj_name(self.ck, layer)
        if name not in self.base:
            self.base[name] = self.ck.tensors[name].read()
        return name, self.base[name], self.ck.tensors[name].dtype

    def analyse(self, layers: list[int]) -> dict[int, Layer]:
        self.rt.reset()
        stats = collect(self.rt, self.spans, layers)
        out = {}
        for i in layers:
            _, bits, dtype = self._bits(i)
            u, cols, write = writers(bits, dtype, stats[i], self.k)
            out[i] = Layer(i, u, cols, write)
        return out

    def tensors(self, plan: list[Layer], alpha: float) -> dict[str, tuple[np.ndarray, str]]:
        out = {}
        for l in plan:
            name, bits, dtype = self._bits(l.index)
            out[name] = (project_columns(bits, dtype, l.direction, l.columns, alpha).reshape(bits.shape), dtype)
        return out

    def score(self, plan: list[Layer], alpha: float) -> float:
        self.rt.set_tensors(self.tensors(plan, alpha))
        value = self.probe().value
        self.rt.reset()
        return value

    def search(self, layers: list[int], alphas=(1, 2, 4, 6, 8), pick: int = 2) -> Plan:
        """Rank layers singly, then add them greedily while the score improves."""
        found = self.analyse(layers)
        trace = []
        single = []
        for i in layers:
            best = max((self.score([found[i]], a), a) for a in alphas)
            single.append((best[0], i, best[1]))
            trace.append({"layer": i, "alpha": best[1], "score": best[0]})
        single.sort(reverse=True)

        chosen = [found[single[0][1]]]
        best_score, best_alpha = single[0][0], single[0][2]
        for _, i, _ in single[1:]:
            if len(chosen) >= pick:
                break
            trial = chosen + [found[i]]
            value, alpha = max((self.score(trial, a), a) for a in alphas)
            trace.append({"layers": [l.index for l in trial], "alpha": alpha, "score": value})
            if value > best_score:
                chosen, best_score, best_alpha = trial, value, alpha
        return Plan(chosen, best_alpha, best_score, trace)

    def patch(self, plan: Plan, meta: dict | None = None) -> Patch:
        edits = []
        for name, (bits, dtype) in self.tensors(plan.layers, plan.alpha).items():
            edits.append(diff_tensor(name, dtype, self.base[name], bits))
        return Patch(edits, {"method": "difference of means, column projection",
                             "layers": [l.index for l in plan.layers], "alpha": plan.alpha,
                             "columns": self.k, **(meta or {})})


def matched_random(plan: Plan, seed: int = 0) -> list[Layer]:
    """Same layers, same number of columns, same strength, random orthogonal direction."""
    rng = np.random.default_rng(seed)
    out = []
    for l in plan.layers:
        v = rng.standard_normal(l.direction.shape)
        v -= (v @ l.direction) * l.direction
        out.append(Layer(l.index, v / np.linalg.norm(v), l.columns, l.write))
    return out


