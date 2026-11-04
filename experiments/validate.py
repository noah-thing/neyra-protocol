"""Generate on fresh problems with several variants of the model, same seeds, and compare."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from lesion.behavior import score
from lesion.interop import from_weight_space_logs
from lesion.patch import Patch
from lesion.runtime import Sampling
from lesion.runtime.mlx import MlxRuntime
from lesion.stats import paired
from lesion.store import Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--variant", action="append", default=[], help="name=path, path to .lesion or edit dir")
ap.add_argument("--practice", required=True)
ap.add_argument("--per-source", type=int, default=32)
ap.add_argument("--skip", type=int, default=32, help="problems per source already used for fitting")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--seed", type=int, default=5678)
ap.add_argument("-o", "--out", default="validate.json")
a = ap.parse_args()

rt = MlxRuntime(a.model)
ck = Checkpoint.open(a.model)
rng = random.Random(a.seed)
used = random.Random(1234)
items = []
for name in ("gsm8k_train", "math_train"):
    rows = [json.loads(x) for x in open(f"{a.practice}/{name}.jsonl")]
    taken = {r["id"] for r in used.sample(rows, a.skip)}
    items += rng.sample([r for r in rows if r["id"] not in taken], a.per_source)

suffix = " Please reason step by step, and put your final answer within \\boxed{}."
prompts = [rt.encode_chat(p["problem"] + suffix) for p in items]
sampling = Sampling(0.6, 0.95, 20, 1.5, 7)


def load(path: str) -> Patch:
    return Patch.load(path) if path.endswith(".lesion") else from_weight_space_logs(path)


results = {}
for spec in ["base="] + a.variant:
    name, path = spec.split("=", 1)
    rt.reset()
    if path:
        p = load(path)
        rt.set_tensors({e.name: (e.apply(ck.tensors[e.name].read(), verify=False), e.dtype) for e in p.edits})
    t = time.time()
    rows = rt.generate(prompts, a.max_tokens, sampling)
    o = [score(r.text, r.hit_limit, i["answer"]) for r, i in zip(rows, items)]
    results[name] = {
        "loop": [float(x.repeated >= 0.25) for x in o], "repeated": [x.repeated for x in o],
        "closed": [float(x.closed) for x in o], "correct": [float(bool(x.correct)) for x in o],
        "finished": [float(x.finished) for x in o],
    }
    print(name, {k: round(float(np.mean(v)), 3) for k, v in results[name].items()}, round(time.time() - t), "s", flush=True)
    Path(a.out).write_text(json.dumps(results))

for name, r in results.items():
    if name == "base":
        continue
    d = paired(np.array(results["base"]["repeated"]), np.array(r["repeated"]))
    print(name, "repeated share vs base", round(d.difference, 3), [round(d.low, 3), round(d.high, 3)], flush=True)
