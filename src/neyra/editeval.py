"""Did an edit take, did it spread, and what did it cost.

The first four numbers are the CounterFact measures of Meng et al. (2022). Efficacy, paraphrase and
neighbourhood score are the share of prompts where the new object beats the old one, or for
neighbours the old one still beats the new. Fluency is the weighted bigram and trigram entropy of
greedy continuations, and damage is the KL divergence from the unedited model on neutral text.
Each is measured before the edit too, because a neighbourhood the model already got wrong is not
the edit's fault.
"""

from __future__ import annotations

import re
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


def words_of(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def appears(text: str, obj: str) -> bool:
    """Whether the object shows up as whole words in a generated continuation, ignoring case and punctuation."""
    needle = words_of(obj)
    return bool(needle) and f" {needle} " in f" {words_of(text)} "


def first_sentence(text: str) -> str:
    """The text up to the first full stop or line break, where EasyEdit's decoding script stops generating."""
    return re.split(r"\.|\n", text, maxsplit=1)[0]


def live_scores(target: str, true: str | None, spoken: dict, stop: bool = False) -> dict[str, float | None]:
    """Live decoding scores of one fact (The Mirage of Model Editing, ACL 2025).

    The probability comparison above asks whether the new object is the likelier of two given ones. This asks
    what the model says when it is left to decode: no teacher forcing, and no cut at the length of the target.
    `spoken` holds the greedy continuations: "edit" (one text), "para", and optionally "near" (prompts about
    other subjects, whose answer should stay `true`) and "gen" (open prompts about the subject). With `stop`
    every text is cut at its first sentence, as EasyEdit does.
    """
    def share(obj, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [first_sentence(t) for t in texts or []] if stop else texts or []
        return float(np.mean([appears(t, obj) for t in texts])) if obj and texts else None

    return {"efficacy": share(target, spoken["edit"]), "paraphrase": share(target, spoken["para"]),
            "old_efficacy": share(true, spoken["edit"]), "old_paraphrase": share(true, spoken["para"]),
            "neighbours": share(true, spoken.get("near")), "leak": share(target, spoken.get("near")),
            "free": share(target, spoken.get("gen"))}


def live(rt, words: Words, requests: list[Request], tokens: int = 24, batch: int = 16, wide: bool = False) -> list[dict]:
    """Greedy continuations of the edit prompt and of each paraphrase, one entry per request.

    `wide` adds the first six neighbour prompts (16 tokens) and the first two generation prompts (40 tokens).
    """
    def run(per_request, n):
        prompts, owner = [], []
        for i, texts in enumerate(per_request):
            for p in texts:
                prompts.append(words.ids(p))
                owner.append(i)
        rows = rt.generate(prompts, n, Sampling(0.0), batch=batch) if prompts else []
        out: list[list[str]] = [[] for _ in per_request]
        for i, row in zip(owner, rows):
            out[i].append(row.text)
        return out

    got = run([(r.text(), *r.paraphrases) for r in requests], tokens)
    out = [{"edit": g[0], "para": g[1:]} for g in got]
    if wide:
        for key, picks, n in (("near", [r.neighbours[:6] for r in requests], 16),
                              ("gen", [r.generation[:2] for r in requests], 40)):
            for o, g in zip(out, run(picks, n)):
                o[key] = g
    return out


def live_summary(requests: list[Request], texts: list[dict], stop: bool = False) -> dict[str, float | None]:
    got = [live_scores(r.target, r.true, t, stop) for r, t in zip(requests, texts)]

    def mean(key):
        vals = [g[key] for g in got if g[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {k: mean(k) for k in got[0]} if got else {}


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
