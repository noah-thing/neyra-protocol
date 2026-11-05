"""Score model directories on the frozen merge benchmark. See candidates/merge_protocol.md.

usage: mm_eval.py SPLIT TASKS MODEL_DIR [MODEL_DIR ...]
  SPLIT  dev | test | all
  TASKS  comma list of gsm8k, mbpp, humaneval, nll
Prints one RESULT line per model with per item outcomes, so the bridge log carries everything.
"""

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DATA = Path(os.path.expanduser("~/merge-exp/data"))
SPLITS = {"gsm8k": (80, 250), "mbpp": (80, 200), "humaneval": (64, 100)}  # (dev, test) item counts
MAX_NEW = {"gsm8k": 256, "mbpp": 200, "humaneval": 320}
STOP_CODE = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert "]
SALT = "neyra-merge-prereg-1"


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def ranked(task, ids):
    return sorted(ids, key=lambda i: hashlib.sha256(f"{SALT}/{task}/{i}".encode()).hexdigest())


def pick(task, ids, split):
    dev, test = SPLITS[task]
    order = ranked(task, ids)
    return {"dev": order[:dev], "test": order[dev:dev + test], "all": order[:dev + test]}[split]


def clean(answer):
    return re.sub(r"<<[^>]*>>", "", answer)


def gsm_items():
    rows = jsonl(DATA / "gsm8k_test.jsonl")
    return {str(i): r for i, r in enumerate(rows)}


def gsm_prompt(shots, q):
    return "".join(f"Question: {s['question']}\nAnswer: {clean(s['answer'])}\n\n" for s in shots) + f"Question: {q}\nAnswer:"


def last_number(text):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m is None:
        found = re.findall(r"-?[\d,]*\.?\d+", text)
        if not found:
            return None
        m = found[-1]
    else:
        m = m.group(1)
    try:
        return float(m.replace(",", ""))
    except ValueError:
        return None


def human_items():
    with gzip.open(DATA / "HumanEval.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    return {r["task_id"]: r for r in rows}


def run_code(program, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.py"
        path.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=d, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def generate(model, tok, prompts, max_new):
    from mlx_lm.generate import batch_generate
    out = []
    for lo in range(0, len(prompts), 24):
        ids = [tok.encode(p) for p in prompts[lo:lo + 24]]
        res = batch_generate(model, tok, ids, max_tokens=max_new)
        out += res.texts
    return out


def score_gsm(model, tok, split):
    items = gsm_items()
    train = jsonl(DATA / "gsm8k_train.jsonl")[:4]
    ids = pick("gsm8k", list(items), split)
    prompts = [gsm_prompt(train, items[i]["question"]) for i in ids]
    texts = generate(model, tok, prompts, MAX_NEW["gsm8k"])
    ok = []
    for i, t in zip(ids, texts):
        t = t.split("\n\nQuestion:")[0].split("Question:")[0]
        want = float(items[i]["answer"].split("####")[1].strip().replace(",", ""))
        got = last_number(t)
        ok.append(int(got is not None and abs(got - want) < 1e-6))
    return ids, ok


def score_human(model, tok, split):
    items = human_items()
    ids = pick("humaneval", list(items), split)
    texts = generate(model, tok, [items[i]["prompt"] for i in ids], MAX_NEW["humaneval"])
    ok = []
    for i, t in zip(ids, texts):
        cut = min([t.find(s) for s in STOP_CODE if s in t] or [len(t)])
        it = items[i]
        program = it["prompt"] + t[:cut] + "\n\n" + it["test"] + f"\ncheck({it['entry_point']})\n"
        ok.append(int(run_code(program)))
    return ids, ok


def mbpp_items():
    rows = jsonl(DATA / "mbpp.jsonl")
    for r in rows:
        r["code"] = r["code"].replace("\r\n", "\n")
        r["test_list"] = [t.replace("\r\n", "\n") for t in r["test_list"]]
    return {str(r["task_id"]): r for r in rows}


def mbpp_prompt(shots, r):
    def head(x):
        return (f"You are an expert Python programmer, and here is your task: {x['text']} "
                f"Your code should pass these tests:\n\n" + "\n".join(x["test_list"]) + "\n[BEGIN]\n")
    return "".join(head(s) + s["code"] + "\n[DONE]\n\n" for s in shots) + head(r)


def score_mbpp(model, tok, split):
    items = mbpp_items()
    shots = [items[str(i)] for i in (2, 3, 4)]
    ids = pick("mbpp", [k for k, r in items.items() if 11 <= r["task_id"] <= 510], split)
    texts = generate(model, tok, [mbpp_prompt(shots, items[i]) for i in ids], MAX_NEW["mbpp"])
    ok = []
    for i, t in zip(ids, texts):
        code = t.split("[DONE]")[0].replace("[BEGIN]", "")
        r = items[i]
        program = r.get("test_setup_code", "") + "\n" + code + "\n\n" + "\n".join(r["test_list"]) + "\n"
        ok.append(int(run_code(program)))
    return ids, ok


def nll_items():
    gsm = jsonl(DATA / "gsm8k_train.jsonl")[-150:]
    code = jsonl(DATA / "mbpp.jsonl")[-150:]
    return ([f"Question: {r['question']}\nAnswer: {clean(r['answer'])}" for r in gsm],
            [f"# {r['text']}\n{r['code']}" for r in code])


def mean_nll(model, tok, texts):
    import mlx.core as mx
    out = []
    for t in texts:
        ids = mx.array(tok.encode(t)[:512])[None]
        logits = model(ids[:, :-1]).astype(mx.float32)
        lse = mx.logsumexp(logits, axis=-1)
        pick_ = mx.take_along_axis(logits, ids[:, 1:, None], axis=-1)[..., 0]
        out.append(float((lse - pick_).mean()))
    return out


def main():
    split, tasks, dirs = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
    from mlx_lm import load
    for d in dirs:
        t0 = time.time()
        model, tok = load(d)
        res = {"model": d, "split": split}
        if "gsm8k" in tasks:
            res["gsm8k_ids"], res["gsm8k"] = score_gsm(model, tok, split)
        if "humaneval" in tasks:
            res["humaneval_ids"], res["humaneval"] = score_human(model, tok, split)
        if "mbpp" in tasks:
            res["mbpp_ids"], res["mbpp"] = score_mbpp(model, tok, split)
        if "nll" in tasks:
            math_texts, code_texts = nll_items()
            res["nll_math"], res["nll_code"] = mean_nll(model, tok, math_texts), mean_nll(model, tok, code_texts)
        res["seconds"] = round(time.time() - t0)
        print("RESULT " + json.dumps(res), flush=True)
        summary = {k: round(sum(v) / len(v), 4) for k, v in res.items() if isinstance(v, list) and v and not k.endswith("ids")}
        print("SUMMARY", d, summary, f"{res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
