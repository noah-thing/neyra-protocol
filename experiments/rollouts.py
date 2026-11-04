"""Generate on a fixed sample of practice problems and save tokens, text and scores."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from neyra.behavior import score
from neyra.interop import from_weight_space_logs
from neyra.lab import read_jsonl
from neyra.patch import Patch
from neyra.runtime import Sampling
from neyra.runtime.mlx import MlxRuntime
from neyra.store import Checkpoint
from neyra.tasks import SUFFIX

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--practice", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--patch", help=".neyra file or a Weight-Space Programming edit folder")
ap.add_argument("--thinking", choices=["on", "off"], default="on")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--per-source", type=int, default=32)
ap.add_argument("--offset", type=int, default=0, help="skip this many sampled problems per source")
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--penalty", type=float, default=1.5)
a = ap.parse_args()

rt = MlxRuntime(a.model)
ck = Checkpoint.open(a.model)
if a.patch:
    p = Patch.load(a.patch) if a.patch.endswith(".neyra") else from_weight_space_logs(a.patch)
    rt.set_tensors({e.name: (e.apply(ck.tensors[e.name].read()), e.dtype) for e in p.edits})

items = []
for source in ("gsm8k_train", "math_train"):
    rows = read_jsonl(f"{a.practice}/{source}.jsonl")
    order = random.Random(a.seed).sample(rows, a.offset + a.per_source)
    items += order[a.offset:]

prompts = [rt.encode_chat(i["problem"] + SUFFIX, a.thinking == "on") for i in items]
t = time.time()
rows = rt.generate(prompts, a.max_tokens, Sampling(0.6, 0.95, 20, a.penalty, 0), a.batch)
seconds = time.time() - t
o = [score(r.text, r.tokens, r.hit_limit, a.max_tokens, i["answer"]) for r, i in zip(rows, items)]
Path(f"{a.name}.json").write_text(json.dumps({
    "config": vars(a), "seconds": seconds,
    "rows": [{"id": i["id"], "answer": i["answer"], "text": r.text, "tokens": r.tokens, "hit_limit": r.hit_limit,
              "correct": bool(x.correct), "looping": x.looping, "tail": x.tail, "closed": x.closed}
             for i, r, x in zip(items, rows, o)],
}))
n = sum(len(r.tokens) for r in rows)
print(a.name, f"n={len(rows)} correct={np.mean([bool(x.correct) for x in o]):.3f} looping={np.mean([x.looping for x in o]):.3f} "
      f"hit_limit={np.mean([r.hit_limit for r in rows]):.3f} closed={np.mean([x.closed for x in o]):.3f} "
      f"mean_tokens={n / len(rows):.0f} {n / seconds:.0f} tok/s {seconds:.0f}s", flush=True)
