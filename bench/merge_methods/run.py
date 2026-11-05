"""Run the merge study steps listed in a spec file, resuming where the last run stopped.

usage: mm_run.py SPEC [BUDGET_SECONDS]
Spec lines (blank lines and # are ignored):
  fetch NAME...                       download public checkpoints
  merge OUT METHOD PARENTS WEIGHTS [key=value ...]    Neyra merge of PARENTS (comma list) from ~/merge-exp/models
  mk OUT METHOD PARENTS WEIGHTS [key=value ...]       the same merge by mergekit
  cmp A B                             bytes of two merged models, in ulps
  eval SPLIT TASKS NAME...            score models in models/ or out/ (see mm_eval.py)
  recheck OUT                         recompute a Neyra merge from its recipe and compare every tensor
  ref OUT PARENTS WEIGHTS             float32 arithmetic in the order given, rounded once
  rm NAME...                          delete models or outputs
A finished step is recorded in ~/merge-exp/state, so a killed run is simply submitted again.
"""

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "mm_src"))
sys.path.insert(0, HERE)

import mm_eval
import mm_fetch
import numpy as np

ROOT = os.environ.get("MM_ROOT", os.path.expanduser("~/merge-exp"))
MODELS, OUT, STATE = f"{ROOT}/models", f"{ROOT}/out", f"{ROOT}/state"
T0 = time.time()


def slurp(path):
    with open(path) as f:
        return f.read()


def say(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def where(name):
    return f"{MODELS}/{name}" if os.path.isdir(f"{MODELS}/{name}") else f"{OUT}/{name}"


def finish(out):
    for f in os.listdir(f"{MODELS}/base"):
        if not f.endswith(".safetensors") and not os.path.exists(f"{out}/{f}"):
            shutil.copy(f"{MODELS}/base/{f}", out)


def params(args):
    out = {}
    for a in args:
        k, v = a.split("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def neyra_parents(names):
    from neyra.methods import CheckpointParent
    from neyra.protocol import root_of, tensor_table
    from neyra.store import Checkpoint
    out = {}
    for n in names:
        c = Checkpoint.open(f"{MODELS}/{n}")
        out[n] = CheckpointParent(c, root_of(tensor_table(c)))
    return out


def do_merge(out, method, parents, weights, *rest):
    from neyra.methods import Recipe, stream
    from neyra.patch import write_checkpoint
    from neyra.store import Checkpoint
    names, w = parents.split(","), [float(x) for x in weights.split(",")]
    base = Checkpoint.open(f"{MODELS}/base")
    ps = neyra_parents(names)
    kw = params(rest)
    if method == "slerp" and "first" in kw:
        kw["first"] = ps[kw["first"]].id
    recipe = Recipe.make(method, [ps[n].id for n in names], w, **kw)
    results, shared = {}, [0, 0]
    for tname, bits, overlap in stream(base, [ps[n] for n in names], recipe):
        results[tname] = bits.reshape(base.tensors[tname].shape)
        shared[0] += overlap.entries
        shared[1] += overlap.disagree
    d = f"{OUT}/{out}"
    shutil.rmtree(d, ignore_errors=True)
    write_checkpoint(base, results, d)
    doc = {"recipe": recipe.to_json(), "recipe_id": recipe.id, "names": dict(zip(names, [ps[n].id for n in names])),
           "merged_tensors": len(results), "shared": shared[0], "differ": shared[1]}
    with open(f"{d}/recipe.json", "w") as f:
        json.dump(doc, f, indent=1)
    say("merge", out, method, names, w, kw, "tensors", len(results), "shared", shared)


def do_recheck(out):
    from neyra.methods import Recipe, stream
    from neyra.protocol import bits_digest
    from neyra.store import Checkpoint
    with open(f"{OUT}/{out}/recipe.json") as f:
        doc = json.load(f)
    recipe = Recipe.from_json(doc["recipe"])
    base, merged = Checkpoint.open(f"{MODELS}/base"), Checkpoint.open(f"{OUT}/{out}")
    ps = neyra_parents(list(doc["names"]))
    order = list(ps)[::-1]
    bad = n = 0
    for tname, bits, _ in stream(base, [ps[k] for k in order], recipe):
        ref = merged.tensors[tname]
        n += 1
        bad += bits_digest(ref.dtype, bits.reshape(ref.shape)) != bits_digest(ref.dtype, ref.read())
    say(f"RECHECK {out}: recomputed {n} tensors from the recipe, parents given in reverse order, {bad} differ")


def do_mk(out, method, parents, weights, *rest):
    names, w = parents.split(","), [float(x) for x in weights.split(",")]
    kw = params(rest)
    name = {"task": "task_arithmetic"}.get(method, method)
    lines = [f"merge_method: {name}", f"base_model: {MODELS}/base", "models:"]
    first = kw.pop("first", None)
    if method == "slerp":
        lines[1] = f"base_model: {MODELS}/{first}"
        names = [n for n in names if n != first]
        lines += [f"  - model: {MODELS}/{n}" for n in names]
        lines += ["parameters:", f"  t: {kw['t']}", "dtype: bfloat16"]
    elif method == "model_stock":
        lines += [f"  - model: {MODELS}/{n}" for n in names] + ["dtype: bfloat16"]
    else:
        for n, x in zip(names, w):
            lines += [f"  - model: {MODELS}/{n}", "    parameters:", f"      weight: {x}"]
            if "density" in kw:
                lines.append(f"      density: {kw['density']}")
        lines += ["parameters:"]
        for k in ("normalize", "lambda", "rescale"):
            if k in kw:
                lines.append(f"  {k}: {str(kw[k]).lower()}")
        lines += ["dtype: float32\nout_dtype: bfloat16" if kw.get("dtype") == "f32" else "dtype: bfloat16"]
    os.makedirs(OUT, exist_ok=True)
    cfg = f"{OUT}/{out}.yaml"
    text = "\n".join(lines) + "\n"
    with open(cfg, "w") as f:
        f.write(text)
    d = f"{OUT}/{out}"
    shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run([f"{ROOT}/venv/bin/mergekit-yaml", cfg, d, "--lazy-unpickle", "--copy-tokenizer"], capture_output=True, text=True, check=False)
    if r.returncode:
        say("mergekit failed", out, r.stderr[-1200:])
        return
    finish(d)
    say("mergekit", out, text.replace("\n", " | "))


def do_ref(out, parents, weights):
    """The plain way: float32 arithmetic in the order given, rounded to bf16 once at the end."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    names, w = parents.split(","), [float(x) for x in weights.split(",")]
    base = safe_open(f"{MODELS}/base/model.safetensors", "pt")
    files = [safe_open(f"{MODELS}/{n}/model.safetensors", "pt") for n in names]
    res = {}
    for k in base.keys():  # noqa: SIM118 - a safetensors handle, not a dict
        b = base.get_tensor(k).float()
        acc = b.clone()
        for f, x in zip(files, w):
            acc = acc + x * (f.get_tensor(k).float() - b)
        res[k] = acc.to(torch.bfloat16).contiguous()
    d = f"{OUT}/{out}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    save_file(res, f"{d}/model.safetensors", metadata={"format": "pt"})
    finish(d)
    say("float32 reference", out, names, w)


def ulp_gap(x, y):
    def key(u):
        u = u.astype(np.int64)
        return np.where(u & 0x8000, 0x8000 - (u & 0x7FFF), u + 0x8000)
    return np.abs(key(x) - key(y))


def do_cmp(a, b):
    import torch
    from safetensors import safe_open
    fa, fb = safe_open(f"{where(a)}/model.safetensors", "pt"), safe_open(f"{where(b)}/model.safetensors", "pt")
    n = diff = tensors = worst = 0
    for k in fa.keys():  # noqa: SIM118 - a safetensors handle, not a dict
        x = fa.get_tensor(k).view(torch.int16).numpy().view(np.uint16)
        y = fb.get_tensor(k).view(torch.int16).numpy().view(np.uint16)
        d = x != y
        n += d.size
        diff += int(d.sum())
        tensors += int(d.any())
        if d.any():
            worst = max(worst, int(ulp_gap(x[d], y[d]).max()))
    say(f"COMPARE {a} vs {b}: {diff:,} of {n:,} entries differ ({diff / n:.3e}), {tensors} tensors, max {worst} ulp")


def do_eval(split, tasks, *names):
    from mlx_lm import load
    for name in names:
        t0 = time.time()
        model, tok = load(where(name))
        res = {"model": name, "split": split}
        tl = tasks.split(",")
        if "gsm8k" in tl:
            res["gsm8k_ids"], res["gsm8k"] = mm_eval.score_gsm(model, tok, split)
        if "humaneval" in tl:
            res["humaneval_ids"], res["humaneval"] = mm_eval.score_human(model, tok, split)
        if "mbpp" in tl:
            res["mbpp_ids"], res["mbpp"] = mm_eval.score_mbpp(model, tok, split)
        if "nll" in tl:
            m, c = mm_eval.nll_items()
            res["nll_math"], res["nll_code"] = mm_eval.mean_nll(model, tok, m), mm_eval.mean_nll(model, tok, c)
        res["seconds"] = round(time.time() - t0)
        os.makedirs(f"{ROOT}/results", exist_ok=True)
        with open(f"{ROOT}/results/{name}.{split}.json", "w") as f:
            json.dump(res, f)
        print("RESULT " + json.dumps(res), flush=True)
        brief = {k: round(sum(v) / len(v), 4) for k, v in res.items() if isinstance(v, list) and v and not k.endswith("ids")}
        say("EVAL", name, split, brief, f"{res['seconds']}s")
        del model


def do_rm(*names):
    for n in names:
        shutil.rmtree(f"{MODELS}/{n}", ignore_errors=True)
        shutil.rmtree(f"{OUT}/{n}", ignore_errors=True)
    say("removed", names)


def do_fetch(*names):
    for n in names:
        if not os.path.isdir(f"{MODELS}/{n}"):
            mm_fetch.snapshot_download(mm_fetch.REPOS[n], local_dir=f"{MODELS}/{n}",
                                       allow_patterns=["*.safetensors", "*.json", "*.txt"])
    say("fetched", names)


STEPS = {"ref": do_ref, "fetch": do_fetch, "merge": do_merge, "mk": do_mk, "cmp": do_cmp, "eval": do_eval, "recheck": do_recheck, "rm": do_rm}


def main():
    spec = sys.argv[1]
    done = set(slurp(STATE).split("\n")) if os.path.exists(STATE) else set()
    lines = [ln.strip() for ln in slurp(spec).split("\n") if ln.strip() and not ln.startswith("#")]
    if sys.argv[2:] == ["--pending"]:
        print(sum(ln not in done for ln in lines))
        return
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 780
    os.makedirs(OUT, exist_ok=True)
    for ln in lines:
        if ln in done:
            continue
        if time.time() - T0 > budget:
            say("out of time, resubmit to continue; next:", ln)
            return
        tries = slurp(f"{STATE}.tries").split("\n").count(ln) if os.path.exists(f"{STATE}.tries") else 0
        with open(f"{STATE}.tries", "a") as f:
            f.write(ln + "\n")
        say("step:", ln, f"(attempt {tries + 1})")
        cmd, *args = ln.split(" #")[0].split()
        try:
            if tries >= 2:
                raise RuntimeError("killed twice before finishing")
            STEPS[cmd](*args)
        except Exception as e:  # noqa: BLE001 - a failed step is logged and skipped so the queue keeps moving
            say("STEP FAILED, skipping:", repr(e)[:300])
        with open(STATE, "a") as f:
            f.write(ln + "\n")
        done.add(ln)
    say("spec finished")


if __name__ == "__main__":
    main()
