"""Same direction, same layers, different choice of row vector c in W' = W - u c^T."""

import argparse
import json
from pathlib import Path

import numpy as np

from lesion.interop import from_weight_space_logs
from lesion.lab import neutral_texts
from lesion.numerics import to_float
from lesion.patch import Patch, diff_tensor
from lesion.probe import SelectivityProbe, loop_spans
from lesion.recover import recover
from lesion.repair import apply_row, moments, neutral_energy, selective_row
from lesion.runtime.mlx import MlxRuntime
from lesion.stats import kl_from_logits
from lesion.store import Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("edit_dir")
ap.add_argument("rollouts")
ap.add_argument("--out", default="compare_edits.json")
ap.add_argument("--save", default="patches")
a = ap.parse_args()

rt = MlxRuntime(a.model)
ck = Checkpoint.open(a.model)
rows = json.loads(Path(a.rollouts).read_text())
ids = [rt.tokenizer.encode(r["text"], add_special_tokens=False) for r in rows]
spans = loop_spans(ids, 1024, minimum=48)
half = len(spans) * 2 // 3
fit, held = spans[:half], spans[half:]
print(len(fit), "fit texts", len(held), "held out texts", flush=True)
probe = SelectivityProbe(rt, held)

texts = neutral_texts()
encoded = [(k, rt.tokenizer.encode(t)) for k, v in texts.items() for t in v]


def neutral_logits():
    return [(k, rt.logits_at(x, np.arange(max(2, len(x) // 4), len(x), 2))) for k, x in encoded]


rt.reset()
reference = neutral_logits()

astra = from_weight_space_logs(a.edit_dir)
info = {}
for e in astra.edits:
    layer = int(e.name.split("layers.")[1].split(".")[0])
    base = ck.tensors[e.name].read()
    edited = e.apply(base)
    s = recover(e.name, base, edited, e.dtype)
    w = to_float(base, e.dtype).astype(np.float64)
    c1, c0 = moments(rt, fit, layer)
    u = s.direction
    mask = np.zeros(w.shape[1])
    mask[s.support] = 1
    info[e.name] = dict(base=base, edited=edited, dtype=e.dtype, w=w, u=u, c1=c1, c0=c0,
                        support=s.support, astra_row=s.alpha * mask * (u @ w), alpha=s.alpha)
print("moments ready", flush=True)


def evaluate(name, tensors, rows_by_tensor=None):
    rt.set_tensors(tensors)
    target, keep = probe.drops()
    now = neutral_logits()
    rt.reset()
    kl = {}
    for (k, p), (_, q) in zip(reference, now):
        kl.setdefault(k, []).append(float(kl_from_logits(p, q).mean()))
    per = {k: float(np.mean(v)) for k, v in kl.items()}
    energy = None
    if rows_by_tensor:
        energy = float(np.mean([neutral_energy(r, info[n]["c0"]) for n, r in rows_by_tensor.items()]))
    out = {"name": name, "target_drop": float(target.mean()), "keep_drop": float(keep.mean()),
           "selectivity": float((target - keep).mean()), "neutral_kl": float(np.mean(list(per.values()))),
           "by_domain": per, "neutral_energy": energy}
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items() if k != "by_domain"}), flush=True)
    return out


results = [evaluate("astra", {n: (d["edited"], d["dtype"]) for n, d in info.items()},
                    {n: d["astra_row"] for n, d in info.items()})]
Path(a.save).mkdir(exist_ok=True)

for columns in (None, 64):
    for lam in (0.1, 1, 10, 100):
        for scale in (1, 2, 4, 6):
            rows_ = {}
            for n, d in info.items():
                cols = None if columns is None else d["support"]
                rows_[n] = scale * selective_row(d["w"], d["u"], d["c1"], d["c0"], lam, cols)
            tensors = {n: (apply_row(info[n]["base"], info[n]["dtype"], info[n]["u"], r), info[n]["dtype"]) for n, r in rows_.items()}
            name = f"selective cols={columns or 'all'} lam={lam} x{scale}"
            results.append(evaluate(name, tensors, rows_))
            Patch([diff_tensor(n, info[n]["dtype"], info[n]["base"], t[0]) for n, t in tensors.items()],
                  {"method": name}).save(f"{a.save}/{name.replace(' ', '_').replace('=', '-')}.lesion")
            Path(a.out).write_text(json.dumps(results, indent=1))
