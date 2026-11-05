"""Edit CounterFact facts one batch at a time and score them after every batch.

    python bench/edit_counterfact.py MODEL counterfact.json --n 1000 --batch 100 --solver protect --layers 8,9,10,11,12

CounterFact (Meng et al., 2022) is https://rome.baulab.info/data/dsets/counterfact.json. The first --n cases are
used in file order, as ROME, MEMIT and AlphaEdit do. The last 400 records are the tuning pool and are never
taken by --n below 21,500. With --single each case is edited alone on the unedited model and the scores are
averaged, which is the table ROME reports. Needs the MLX backend.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from neyra.edit import (
    Editor,
    Request,
    Settings,
    Words,
    load_requests,
    neutral_windows,
    second_moments,
)
from neyra.editeval import Neutral, flags, probe, summary
from neyra.runtime import open_runtime
from neyra.store import Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("data")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--batch", type=int, default=100)
ap.add_argument("--layers", required=True)
ap.add_argument("--solver", default="protect", choices=["protect", "memit", "null"])
ap.add_argument("--lam", type=float, default=5000.0)
ap.add_argument("--text", help="plain text for the covariance, such as wikitext")
ap.add_argument("--single", action="store_true")
ap.add_argument("--out", default="edit_counterfact.json")
a = ap.parse_args()

TUNING_POOL = 400
requests = load_requests(a.data)
assert a.n <= len(requests) - TUNING_POOL, "the last records are for tuning only"
requests = requests[:a.n]
layers = [int(x) for x in a.layers.split(",")]
rt = open_runtime(a.model, "mlx")
ck = Checkpoint.open(a.model)
words = Words(rt.tokenizer)
settings = Settings(layers=layers, solver=a.solver, lam=a.lam, batch=1 if a.single else a.batch)

windows = neutral_windows(rt, words, a.text, 256 + 32, 96, 0)
fit, held = windows[:-32], windows[-32:]
cov = second_moments(rt, fit, layers)
reference = Neutral(rt, held)
before = summary(probe(rt, words, requests))
print("before", json.dumps(before), flush=True)

result = {"model": a.model, "n": a.n, "settings": vars(settings), "before": before}
start = time.time()
if a.single:
    rows = []
    for r in requests:
        rt.reset()
        editor = Editor(rt, ck, settings, layers, cov, say=lambda *_: None)
        editor.write([r])
        f = flags(probe(rt, words, [r]))
        rows.append([float(f[k].mean()) for k in ("efficacy", "paraphrase", "specificity")])
    es, ps, ns = np.mean(rows, axis=0)
    result["after"] = {"efficacy": es, "paraphrase": ps, "specificity": ns, "score": 3 / (1 / es + 1 / ps + 1 / ns)}
else:
    editor = Editor(rt, ck, settings, layers, cov, say=lambda *_: None)
    curve = []
    for i in range(0, len(requests), a.batch):
        editor.write(requests[i:i + a.batch])
        now = summary(probe(rt, words, [Request(r.subject, r.prompt, r.target, r.true) for r in requests[:editor.written]]))
        curve.append({"edits": editor.written, "efficacy": now["efficacy"],
                      "neutral_kl": float(reference.drift(rt)["kl"].mean())})
        print(json.dumps(curve[-1]), flush=True)
    result["curve"], result["after"] = curve, summary(probe(rt, words, requests))
result["seconds"] = round(time.time() - start, 1)
Path(a.out).write_text(json.dumps(result, indent=2, default=float))
print(json.dumps(result["after"], default=float))
