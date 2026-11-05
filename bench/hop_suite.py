"""Score `neyra hop` on the planted failures, and show how today's `check` and `diff` do on the same files.

    python bench/hop_suite.py            # the score of neyra hop
    python bench/hop_suite.py --today    # the same files through neyra check and neyra diff

A case passes when the first sentence names the artifact that first differs and the cause is the planted one.
A control passes when the tool finds nothing wrong, which is how false alarms are counted.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from hop_cases import CASES, CONTROLS, Case

from neyra.cli import main
from neyra.hop import document, trace

# What a person reading the output of `neyra check` and `neyra diff` would take as the cause. Read off the
# real output for each case; a case with no pattern is one where neither command says anything about it.
TODAY = {
    "merge_wrote_base": r"NEVER APPLIED",
    "delta_rounds_away": r"rounding kept under half",
    "partial_merge": r"were not changed",
    "quant_config_left": r"quantization_config",
    "wrong_layer_range": r"only in the merged model",
    "dtype_ignored": r"stored in F32",
    "truncated": r"missing: not in the merged",
    "template_missing": r"chat template: removed",
    "lm_head_lost": r"lm_head.weight\s+missing",
    "template_changed": r"chat template: \d+ lines differ",
    "tokenizer_changed": r"tokenizer.json differs",
}


def run_cli(argv: list[str]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            main(argv)
        except SystemExit:
            pass
    return buf.getvalue()


def today(case: Case, paths: list[Path]) -> tuple[bool, bool]:
    base, adapter, merged = paths[:3]
    text = run_cli(["check", str(base), str(merged), "--adapter", str(adapter)]) + "\n" + run_cli(["diff", str(base), str(merged)])
    first_line = text.splitlines()[0] if text else ""
    pattern = TODAY.get(case.name)
    return bool(pattern and re.search(pattern, first_line)), bool(pattern and re.search(pattern, text))


def hop_result(case: Case, paths: list[Path]) -> tuple[bool, str]:
    doc = document(trace([str(p) for p in paths]))
    first = doc["first"]
    if case.cause is None:
        return first is None, doc["verdict"]
    if first is None:
        return False, doc["verdict"]
    right = first["role"] == case.first and first["cause"] == case.cause
    names_hop = re.search(rf"\b{re.escape(case.first)}\b|{'merged model' if case.first == 'merged' else 'GGUF'}",
                          doc["verdict"].split(".")[0]) is not None
    return right and names_hop, doc["verdict"]


def build(case: Case, root: Path, seed: int = 1) -> list[Path]:
    return case.build(root, np.random.default_rng(seed))


def score(show: bool = True) -> dict:
    tally = {"kill": [0, 0], "gguf": [0, 0], "controls": [0, 0]}
    for case in CASES + CONTROLS:
        key = "controls" if case.cause is None else "kill" if case.kill_set else "gguf"
        with tempfile.TemporaryDirectory() as root:
            ok, sentence = hop_result(case, build(case, Path(root)))
        tally[key][0] += ok
        tally[key][1] += 1
        if show:
            print(f"{'pass' if ok else 'FAIL'}  {case.name:<22} {sentence}")
    if show:
        for label, (hit, total) in tally.items():
            print(f"{label:<9} {hit} of {total}")
    return tally


def score_today() -> None:
    line = anywhere = 0
    kill = [c for c in CASES if c.kill_set]
    for case in kill:
        with tempfile.TemporaryDirectory() as root:
            a, b = today(case, build(case, Path(root)))
        line += a
        anywhere += b
        print(f"{case.name:<24} first line: {'yes' if a else 'no':<4} anywhere in check and diff: {'yes' if b else 'no'}")
    print(f"first line names the cause: {line} of {len(kill)}")
    print(f"cause appears somewhere in the output of both commands: {anywhere} of {len(kill)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--today", action="store_true")
    args = ap.parse_args()
    if args.today:
        score_today()
    else:
        tally = score()
        sys.exit(0 if all(h == t for h, t in tally.values()) else 1)
