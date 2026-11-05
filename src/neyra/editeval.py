"""Did an edit take, did it spread, and what did it cost.

The first four numbers are the CounterFact measures of Meng et al. (2022). Efficacy, paraphrase and
neighbourhood score are the share of prompts where the new object beats the old one, or for
neighbours the old one still beats the new. Fluency is the weighted bigram and trigram entropy of
greedy continuations, and damage is the KL divergence from the unedited model on neutral text.
Each is measured before the edit too, because a neighbourhood the model already got wrong is not
the edit's fault.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .edit import Request, Words
from .runtime import Sampling
from .stats import Paired, paired


@dataclass(frozen=True)
class Limits:
    """What counts as success. Fixed before any run, and recorded in the receipt."""

    efficacy: float = 0.9
    specificity_drop: float = 0.05  # fall in the share of neighbours on the right side
    neutral_kl: float = 0.05  # nats per token
    fluency: float = 0.9  # of the unedited model's entropy


def probe(rt, words: Words, requests: list[Request]) -> dict[str, np.ndarray]:
    """Per prompt negative log probability of the objects, summed over their tokens, in the order of the requests.

    NaN where the old object is unknown.
    """
    rows, entries = [], []
    for i, r in enumerate(requests):
        for kind, prompts in (("edit", (r.text(),)), ("para", r.paraphrases), ("near", r.neighbours)):
            for p in prompts:
                prefix = words.ids(p)
                new, old = len(rows), None
                rows.append((prefix, words.target(p, r.target)))
                if r.true:
                    old = len(rows)
                    rows.append((prefix, words.target(p, r.true)))
                entries.append((kind, i, new, old))
    nll, exact = rt.score(rows) if rows else (np.empty(0), np.empty(0, bool))
    nll = nll * np.array([len(t) for _, t in rows])  # whole object, not per token, as the benchmark compares them
    out = {}
    for kind in ("edit", "para", "near"):
        mine = [e for e in entries if e[0] == kind]
        out[kind + "_owner"] = np.array([e[1] for e in mine], int)
        out[kind + "_new"] = np.array([nll[e[2]] for e in mine])
        out[kind + "_old"] = np.array([nll[e[3]] if e[3] is not None else np.nan for e in mine])
        out[kind + "_exact"] = np.array([exact[e[2]] for e in mine], bool)
    return out


def flags(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """1.0 where a prompt is on the right side, NaN where the old object is unknown."""
    def side(new, old, wanted):
        return np.where(np.isnan(old), np.nan, (new < old if wanted == "new" else old < new).astype(float))

    return {"efficacy": side(p["edit_new"], p["edit_old"], "new"),
            "paraphrase": side(p["para_new"], p["para_old"], "new"),
            "specificity": side(p["near_new"], p["near_old"], "old"),
            "exact": p["edit_exact"].astype(float)}


def summary(p: dict[str, np.ndarray]) -> dict[str, float | int | None]:
    out = {k: (float(v.mean()) if len(v) and not np.isnan(v).any() else None) for k, v in flags(p).items()}
    parts = [out[k] for k in ("efficacy", "paraphrase", "specificity")]
    out["score"] = len(parts) / sum(1 / max(x, 1e-9) for x in parts) if None not in parts else None
    gap = p["edit_old"] - p["edit_new"]
    out["margin"] = float(gap.mean()) if len(gap) and not np.isnan(gap).any() else None
    out.update(n_edit=len(p["edit_new"]), n_para=len(p["para_new"]), n_near=len(p["near_new"]))
    return out


def ngram_entropy(tokens: list[int]) -> float:
    values, weights = [], []
    for n, w in ((2, 2 / 3), (3, 4 / 3)):
        grams = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        if grams:
            f = np.array(list(grams.values()), float)
            f /= f.sum()
            values.append(float(-(f * np.log2(f)).sum()))
            weights.append(w)
    return float(np.average(values, weights=weights)) if values else 0.0


def fluency(rt, words: Words, requests: list[Request], tokens: int = 48, limit: int = 32) -> float:
    prompts = [r.generation[0] if r.generation else r.text() for r in requests[:limit]]
    rows = rt.generate([words.ids(p) for p in prompts], tokens, Sampling(0.0), batch=16)
    return float(np.mean([ngram_entropy(r.tokens) for r in rows]))


class Neutral:
    """The unedited model's next token distributions on held out text, kept as their top entries."""

    def __init__(self, rt, windows: list[list[int]], keep: int = 64):
        self.windows, self.keep, self.ref = windows, keep, []
        for w in windows:
            lp = self._logprobs(rt, w)
            top = np.argpartition(-lp, keep - 1, axis=1)[:, :keep]
            self.ref.append((top, np.take_along_axis(lp, top, 1), self._actual(lp, w)))

    @staticmethod
    def _logprobs(rt, w: list[int]) -> np.ndarray:
        z = rt.logits_at(w, np.arange(1, len(w))).astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    @staticmethod
    def _actual(lp: np.ndarray, w: list[int]) -> np.ndarray:
        return -lp[np.arange(len(w) - 1), np.asarray(w[1:])]

    def drift(self, rt) -> dict[str, np.ndarray]:
        kl, nll = [], []
        for w, (top, lp, actual) in zip(self.windows, self.ref):
            now = self._logprobs(rt, w)
            kl.append(float((np.exp(lp) * (lp - np.take_along_axis(now, top, 1))).sum(axis=1).mean()))
            nll.append(float((self._actual(now, w) - actual).mean()))
        return {"kl": np.array(kl), "nll": np.array(nll)}


def statements(before: dict, after: dict) -> list[tuple[str, Paired]]:
    """Paired change per measure, over the same prompts."""
    a, b = flags(before), flags(after)
    return [(k, paired(a[k], b[k])) for k in ("efficacy", "paraphrase", "specificity")
            if len(a[k]) > 1 and not np.isnan(a[k]).any()]


def judge(before: dict, after: dict, limits: Limits, drift: dict, fluent_before: float,
          fluent_after: float) -> dict[str, bool]:
    """Pass or fail per gate. A measure that could not be taken fails, because nothing was shown."""
    b, a = summary(before), summary(after)
    gates = {
        "efficacy": a["efficacy"] is not None and a["efficacy"] >= limits.efficacy,
        "specificity": (a["specificity"] is not None and b["specificity"] is not None
                        and b["specificity"] - a["specificity"] <= limits.specificity_drop),
        "neutral_drift": float(drift["kl"].mean()) <= limits.neutral_kl,
        "fluency": fluent_before > 0 and fluent_after >= limits.fluency * fluent_before,
    }
    return gates
