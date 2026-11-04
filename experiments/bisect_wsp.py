"""Which of the 128 edited columns in the published Qwen3.5-0.8B edit do the work?

Searches on teacher forced base model loops, then confirms once on loops it never saw.
"""

import argparse
import json
import time
from pathlib import Path

from neyra.interop import from_weight_space_logs
from neyra.runtime.mlx import MlxRuntime
from neyra.store import Checkpoint
from neyra.workflows import bisect_loops

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("edit_dir")
ap.add_argument("rollouts", help="JSON list of rows with tokens")
ap.add_argument("-o", "--out", default="bisect_wsp.json")
ap.add_argument("--texts", type=int, default=18)
ap.add_argument("--retain", type=float, default=0.9)
a = ap.parse_args()

t0 = time.time()
rt = MlxRuntime(a.model)
ck = Checkpoint.open(a.model)
rows = json.loads(Path(a.rollouts).read_text())
tokens = [r["tokens"] for r in rows if r["looping"]][:a.texts]
patch = from_weight_space_logs(a.edit_dir, ck)
b = bisect_loops(rt, ck, patch, tokens, retain=a.retain, budget=700)
b.patch.save(Path(a.out).with_suffix(".neyra"))
r, c = b.result, b.check
Path(a.out).write_text(json.dumps({
    "units": [str(b.units.units[i]) for i in r.units], "total": r.total, "search_retained": r.retained,
    "minimal": r.minimal, "baseline": r.baseline.value, "full": r.full.value, "achieved": r.achieved.value,
    "weights_full": b.units.weights_changed(range(len(b.units))), "weights_kept": b.units.weights_changed(r.units),
    "confirmed": c.passed, "confirmed_retained": c.retained, "confirm_low": c.low, "confirm_items": c.n,
    "ranking": [[str(b.units.units[i]), s] for i, s in r.ranking], "trials": len(r.trials),
    "seconds": time.time() - t0, "patch_id": b.patch.id,
}, indent=1))
print(f"{len(r.units)} of {r.total} units, confirmed={c.passed} retained={c.retained:.3f} on {c.n} unseen texts, "
      f"search retained {r.retained:.3f}, minimal={r.minimal}, {time.time() - t0:.0f}s", flush=True)
