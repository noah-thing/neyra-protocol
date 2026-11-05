"""How often does a model's declared base agree with its tensors?

Draws public models at random times over the last year, keeps those that declare a base with
`base_model:finetune:X`, and measures each against that base with a few megabytes of range reads.
Writes one JSON line per model. The verdict is decided afterwards from the raw numbers, so the
thresholds cannot be tuned on this sample.

    python benchmarks/declared_base.py --seed 1105 --draws 90 --out declared_base.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from neyra.identity import measure, same_files
from neyra.store import Checkpoint

API = "https://huggingface.co/api/models"
UA = {"User-Agent": "neyra-research (read-only, 1 request a second)"}
FIELDS = ["safetensors", "gated", "tags", "baseModels", "createdAt", "downloads", "library_name"]


def get(url: str):
    time.sleep(1.0)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def window(moment: datetime) -> list[dict]:
    """The 1000 models created just before `moment`."""
    oid = f"{int(moment.timestamp()):08x}" + "0" * 16
    cursor = base64.b64encode(json.dumps({"_id": {"$lt": oid}}, separators=(",", ":")).encode()).decode()
    query = [("limit", "1000"), ("sort", "createdAt"), ("direction", "-1"), ("cursor", cursor)]
    return get(API + "?" + urllib.parse.urlencode(query + [("expand[]", f) for f in FIELDS]))


def usable(m: dict) -> bool:
    declared = m.get("baseModels") or {}
    return (declared.get("relation") == "finetune" and not m.get("gated") and m.get("safetensors")
            and len(declared.get("models", [])) == 1 and "gguf" not in m.get("tags", []))


def card(repo: str) -> dict:
    """What a reader would check before calling a mismatch wrong: the base's own parent, and open discussions."""
    out = {}
    try:
        out["discussions"] = [d["title"] for d in get(f"{API}/{repo}/discussions")["discussions"]][:5]
    except Exception as e:  # noqa: BLE001 - any failure is recorded in the row
        out["discussions"] = f"unavailable: {e}"
    return out


def config_signature(ck: Checkpoint) -> dict:
    raw = ck.text("config.json")
    cfg = json.loads(raw) if raw else {}
    cfg = cfg.get("text_config", cfg)
    keys = ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size")
    return {k: cfg.get(k) for k in keys}


def examine(repo: str, base_repo: str) -> dict:
    row: dict = {"model": repo, "base": base_repo}
    try:
        model, base = Checkpoint.open(repo), Checkpoint.open(base_repo)
    except Exception as e:  # noqa: BLE001 - any failure is recorded in the row
        row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return row
    m = measure(base, model)
    row.update(
        same_files=same_files(base, model),
        config_equal=config_signature(base) == config_signature(model),
        names_equal=sorted(base.tensors) == sorted(model.tensors),
        shapes_equal=({n: r.shape for n, r in base.tensors.items()} == {n: r.shape for n, r in model.tensors.items()}),
        cos=m.cos, rel=m.rel, unchanged=m.unchanged, matrices=len(m.matrices), spans=len(m.probes),
        only_base=len(m.only_base), only_model=len(m.only_model), skipped=len(m.skipped),
        cos_each=[round(p.cos, 4) for p in m.matrices],
        mb_read=round((base.bytes_read + model.bytes_read) / 1e6, 2),
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--draws", type=int, default=90)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    end = datetime(2026, 11, 4)
    with open(a.out, "w") as out:
        for i in range(a.draws):
            moment = end - timedelta(seconds=rng.randrange(a.days * 86400))
            pool = [m for m in window(moment) if usable(m)]
            if not pool:
                print(i, "empty window", file=sys.stderr)
                continue
            pick = rng.choice(pool)
            base = pick["baseModels"]["models"][0]["id"]
            row = {"draw": i, "moment": moment.isoformat(), "pool": len(pool), "created": pick["createdAt"],
                   "downloads": pick.get("downloads"), **examine(pick["id"], base)}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(i, row["model"], "<-", row["base"], row.get("cos"), row.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
