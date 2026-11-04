"""Rebuild the loop fix with no frontier model, then compare it to the published edit.

Direction, columns, layers and strength all come from measurements on base model
rollouts. Astra's edit is only read afterwards, to score agreement.
"""

import argparse
import json
import time
from pathlib import Path

from lesion.interop import from_weight_space_logs
from lesion.probe import loop_spans
from lesion.recover import recover
from lesion.repair import Repair, down_proj_name
from lesion.runtime.mlx import MlxRuntime
from lesion.store import Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("edit_dir")
ap.add_argument("rollouts")
ap.add_argument("-o", "--out", default="fix.lesion")
ap.add_argument("--report", default="fix_report.json")
ap.add_argument("--columns", type=int, default=64)
a = ap.parse_args()

t0 = time.time()
rt = MlxRuntime(a.model)
ck = Checkpoint.open(a.model)
rows = json.loads(Path(a.rollouts).read_text())
ids = [rt.tokenizer.encode(r["text"], add_special_tokens=False) for r in rows]
spans = loop_spans(ids, 1024, minimum=48)
print(len(spans), "texts with repetition", flush=True)
half = len(spans) * 2 // 3
fit, probe = spans[:half], spans[half:]

repair = Repair(rt, ck, fit, probe, a.columns)
layers = list(range(len(rt.layers())))
plan = repair.search(layers)
print("chosen", [l.index for l in plan.layers], "alpha", plan.alpha, "score", plan.score,
      round(time.time() - t0), "s", flush=True)

patch = repair.patch(plan)
patch.save(a.out)

astra = from_weight_space_logs(a.edit_dir)
agreement = {}
found = repair.analyse(sorted({15, 19} | {l.index for l in plan.layers}))
for e in astra.edits:
    layer = int(e.name.split("layers.")[1].split(".")[0])
    base = ck.tensors[e.name].read()
    s = recover(e.name, base, e.apply(base), e.dtype)
    mine = found[layer]
    overlap = len(set(s.support.tolist()) & set(mine.columns.tolist()))
    agreement[layer] = {"cosine": float(abs(s.direction @ mine.direction)),
                        "columns_shared": overlap, "columns": len(s.support)}
print(agreement, flush=True)

Path(a.report).write_text(json.dumps({
    "layers": [l.index for l in plan.layers], "alpha": plan.alpha, "score": plan.score,
    "trace": plan.trace, "agreement": agreement, "seconds": time.time() - t0,
    "tensor": [down_proj_name(ck, l.index) for l in plan.layers],
}, indent=1))
print("done", round(time.time() - t0), "s", flush=True)
