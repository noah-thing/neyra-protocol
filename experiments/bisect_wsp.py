"""Which of the 128 edited columns in the published Qwen3.5-0.8B edit do the work?

Teacher forced search over base model rollouts, so every trial is deterministic.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lesion.bisect import bisect
from lesion.interop import from_weight_space_logs
from lesion.probe import SelectivityProbe, Span, repeated_mask
from lesion.runtime.mlx import MlxRuntime
from lesion.store import Checkpoint
from lesion.units import UnitSet

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("edit_dir")
ap.add_argument("rollouts")
ap.add_argument("-o", "--out", default="bisect_wsp.json")
ap.add_argument("--texts", type=int, default=16)
ap.add_argument("--length", type=int, default=1536)
ap.add_argument("--retain", type=float, default=0.9)
a = ap.parse_args()

rt = MlxRuntime(a.model)
ck = Checkpoint.open(a.model)
units = UnitSet(from_weight_space_logs(a.edit_dir), ck, "column")
print(len(units), "units", flush=True)

rows = json.loads(Path(a.rollouts).read_text())
spans = []
for r in rows:
    ids = rt.tokenizer.encode(r["text"], add_special_tokens=False)
    mask = repeated_mask(ids, 0)
    if mask.sum() < 64:
        continue
    onset = int(np.argmax(mask))
    lo = max(0, onset - a.length // 2)
    t = ids[lo:lo + a.length]
    m = mask[lo:lo + a.length]
    spans.append(Span(t, 1, m, ~m))
spans = spans[:a.texts]
print(len(spans), "texts,", sum(s.target.sum() for s in spans), "target tokens", flush=True)

probe = SelectivityProbe(rt, spans)
t0 = time.time()
count = 0


def measure(chosen):
    global count
    rt.set_tensors(units.tensors(chosen))
    m = probe()
    count += 1
    if count % 20 == 0:
        print(count, "trials", round(time.time() - t0), "s", flush=True)
    return m


r = bisect(len(units), measure, retain=a.retain)
rt.reset()
Path(a.out).write_text(json.dumps({
    "units": [str(units.units[i]) for i in r.units],
    "total": r.total, "retained": r.retained, "minimal": r.minimal,
    "baseline": r.baseline.value, "full": r.full.value, "achieved": r.achieved.value,
    "weights_full": units.weights_changed(range(len(units))),
    "weights_kept": units.weights_changed(r.units),
    "ranking": [[str(units.units[i]), s] for i, s in r.ranking],
    "trials": len(r.trials), "seconds": time.time() - t0,
}, indent=1))
print(f"{len(r.units)} of {r.total} units keep {r.retained:.3f} of the effect, minimal={r.minimal}, "
      f"{len(r.trials)} trials, {time.time() - t0:.0f}s")
