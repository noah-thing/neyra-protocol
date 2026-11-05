"""Tables and paired bootstrap intervals for the merge study. Reads the per model JSON the scorer saved.

usage: mm_analyze.py RESULTS_DIR dev|test NAME [NAME ...] [--ref NAME ...]
The macro score is the mean of the GSM8K and MBPP accuracies. Intervals resample items within task, the same
resample for every model, so a difference between two models is a paired difference.
"""

import json
import sys
from pathlib import Path

import numpy as np

TASKS = ("gsm8k", "mbpp")
DRAWS = 10_000


def load(d, name, split):
    return json.load(open(Path(d) / f"{name}.{split}.json"))


def table(results):
    """Per model, per task, a 0/1 vector aligned by item id."""
    ids = {t: results[next(iter(results))][f"{t}_ids"] for t in TASKS}
    out = {}
    for name, r in results.items():
        for t in TASKS:
            assert r[f"{t}_ids"] == ids[t], (name, t)
        out[name] = {t: np.array(r[t], dtype=float) for t in TASKS}
    return out


def resamples(table, seed=20261105):
    rng = np.random.default_rng(seed)
    n = {t: len(next(iter(table.values()))[t]) for t in TASKS}
    return {t: rng.integers(0, n[t], size=(DRAWS, n[t])) for t in TASKS}


def macro_draws(scores, idx):
    return np.mean([scores[t][idx[t]].mean(axis=1) for t in TASKS], axis=0)


def interval(x):
    return np.percentile(x, [2.5, 97.5])


def report(d, split, names, refs):
    tab = table({n: load(d, n, split) for n in names})
    idx = resamples(tab)
    rows = []
    for n in names:
        s = tab[n]
        point = np.mean([s[t].mean() for t in TASKS])
        lo, hi = interval(macro_draws(s, idx))
        rows.append((n, s["gsm8k"].mean(), s["mbpp"].mean(), point, lo, hi))
    print(f"| model ({split}) | GSM8K | MBPP | macro | 95% interval |\n|---|---|---|---|---|")
    for n, g, m, p, lo, hi in rows:
        print(f"| {n} | {g:.3f} | {m:.3f} | {p:.3f} | {lo:.3f} to {hi:.3f} |")
    for ref in refs:
        print(f"\nmacro difference against {ref}\n\n| model | difference | 95% interval | above zero |\n|---|---|---|---|")
        base = macro_draws(tab[ref], idx)
        for n in names:
            if n == ref:
                continue
            diff = macro_draws(tab[n], idx) - base
            point = np.mean([tab[n][t].mean() - tab[ref][t].mean() for t in TASKS])
            lo, hi = interval(diff)
            print(f"| {n} | {point:+.3f} | {lo:+.3f} to {hi:+.3f} | {'yes' if lo > 0 else 'no'} |")


if __name__ == "__main__":
    args = sys.argv[1:]
    refs = []
    if "--ref" in args:
        i = args.index("--ref")
        refs = args[i + 1:]
        args = args[:i]
    report(args[0], args[1], args[2:], refs)
