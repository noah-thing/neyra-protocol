"""Calibrate the identity null on PolyPythias and score it on sizes it never saw.

Input: RESULT lines from the panel run (one JSON per pair: `siblings` = two independent runs of one size, `steps` =
a run against its own earlier checkpoint). Pythia 160m sets the thresholds; every other size is held out.

    python bench/identity_panel.py panel_*.log --out src/neyra/identity_null.json

Unrelated pairs are siblings. Derived pairs are a run and an earlier step of itself, at most FAIR_GAP steps behind: the
same weights trained on, which is what a fine-tune is. Larger gaps are reported but not counted as errors.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

CALIBRATION = "160m"
FAIR_GAP = 43000


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.startswith("RESULT"):
                    r = json.loads(line[7:])
                    if r.get("cos") is not None:
                        rows.append(r)
    return rows


def thresholds(rows):
    null = [r["cos"] for r in rows if r["size"] == CALIBRATION and r["kind"] == "siblings"]
    fair = [r["cos"] for r in rows if r["size"] == CALIBRATION and r["kind"] == "steps" and r["gap"] <= FAIR_GAP]
    unrelated = max(0.05, math.ceil((max(null) + 3 * float(np.std(null))) * 100) / 100)
    derived = max(unrelated + 0.1, math.floor(min(fair) * 100) / 100)
    return {"unrelated_max": unrelated, "derived_min": derived}, null, fair


def score(rows, t):
    def call(c):
        return "derived" if c >= t["derived_min"] else "unrelated" if c <= t["unrelated_max"] else "inconclusive"

    table = {}
    for size in sorted({r["size"] for r in rows}):
        sib = [call(r["cos"]) for r in rows if r["size"] == size and r["kind"] == "siblings"]
        pos = [call(r["cos"]) for r in rows if r["size"] == size and r["kind"] == "steps" and r["gap"] <= FAIR_GAP]
        table[size] = {"unrelated_pairs": len(sib), "called_derived": sib.count("derived"), "called_inconclusive": sib.count("inconclusive"),
                       "derived_pairs": len(pos), "called_unrelated": pos.count("unrelated"), "called_inconclusive_derived": pos.count("inconclusive")}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(a.logs)
    t, null, fair = thresholds(rows)
    table = score(rows, t)
    held_sib = sum(v["unrelated_pairs"] for s, v in table.items() if s != CALIBRATION)
    held_fp = sum(v["called_derived"] for s, v in table.items() if s != CALIBRATION)
    held_pos = sum(v["derived_pairs"] for s, v in table.items() if s != CALIBRATION)
    held_fn = sum(v["called_unrelated"] for s, v in table.items() if s != CALIBRATION)
    doc = {**t, "statistic": "median cosine of 16384 entry spans of up to 24 matrices",
           "panel": "EleutherAI PolyPythias: pythia-{14m,31m,70m,160m,410m} and seeds 1-9, final checkpoint; derived = same run at an earlier step",
           "calibrated_on": CALIBRATION, "fair_gap_steps": FAIR_GAP,
           "null": {"pairs": len(null), "mean": float(np.mean(null)), "sd": float(np.std(null)), "max": max(null)},
           "derived_calibration": {"pairs": len(fair), "min": min(fair)},
           "held_out": {"unrelated_pairs": held_sib, "false_derived": held_fp, "false_derived_95ci": wilson(held_fp, held_sib),
                        "derived_pairs": held_pos, "called_unrelated": held_fn, "false_unrelated_95ci": wilson(held_fn, held_pos),
                        "by_size": table},
           "all_gaps": {str(r["size"]) + "@" + str(r["gap"]): round(r["cos"], 3) for r in rows if r["kind"] == "steps"}}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
