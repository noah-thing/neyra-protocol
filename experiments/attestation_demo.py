"""Sign a real edit twice, once as OpenSSF model signing (OMS) and once as a Neyra transformation statement,
check each with its own tools, then reshard the model and see what each one keeps.

    python attestation_demo.py --stock /path/to/venv-with-model-signing [--work DIR]

The stock environment needs model-signing, in-toto-attestation, securesystemslib and jsonschema. Keys are made
in the work directory, which is outside the repository, and are never committed. The model is a 100 KB toy.
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "tests"), str(HERE.parent / "src")]

import tinylm

import neyra.runtime
from neyra import attest as at
from neyra.cli import main
from neyra.edit import Words
from neyra.patch import write_safetensors
from neyra.protocol import root_of, tensor_table
from neyra.store import Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--stock", required=True, help="a virtualenv holding model-signing and the other stock tools")
ap.add_argument("--work", help="directory for everything this writes; default a new temporary one")
a = ap.parse_args()
work = Path(a.work or tempfile.mkdtemp(prefix="neyra-attest-"))
work.mkdir(parents=True, exist_ok=True)
stock = Path(a.stock) / "bin"


def run(*cmd, check=True):
    out = subprocess.run([str(c) for c in cmd], check=False, capture_output=True, text=True)
    if check and out.returncode:
        raise SystemExit(f"{cmd[0]} failed:\n{out.stdout}\n{out.stderr}")
    return out


def oms(*args, check=True):
    return run(stock / "model_signing", *args, check=check)


# 1. A tiny model that knows some facts, and a real `neyra edit` of it.
model, config, known, _ = tinylm.trained("dense")
base = work / "base"
base.mkdir(exist_ok=True)
tinylm.save(model, config, base)
runtime = tinylm.runtime(model)
neyra.runtime.open_runtime = lambda *args, **kw: runtime
words, template = Words(tinylm.Tokenizer()), tinylm.RELATIONS[0][0]
rows = [(words.ids(template.replace("{}", s)), words.target(template, known[(0, s)])) for s in tinylm.SUBJECTS]
right = [s for s, ok in zip(tinylm.SUBJECTS, runtime.score(rows)[1]) if ok]
requests = []
for s in right[:2]:
    old = known[(0, s)]
    requests.append({"subject": s, "prompt": template, "target": next(o for o in tinylm.CAPITALS if o != old),
                     "true": old, "neighbours": [template.replace("{}", o) for o in right[2:] if known[(0, o)] == old]})
(work / "requests.json").write_text(json.dumps(requests))
rng = np.random.default_rng(1)
(work / "neutral.txt").write_text(" ".join(str(rng.choice(tinylm.SUBJECTS + tinylm.CAPITALS)) for _ in range(6000)))
main(["keygen", "-o", str(work / "neyra.key")])
code = main(["edit", str(base), str(work / "requests.json"), "-o", str(work / "edit.neyra"), "--key",
             str(work / "neyra.key"), "--layers", "1,2,3", "--contexts", "0", "--steps", "60", "--lr", "1",
             "--lam", "1000", "--anchor", "30", "--clamp", "30", "--text", str(work / "neutral.txt"),
             "--neutral", "60", "--held", "10", "--window", "8"])
main(["apply", str(base), str(work / "edit.neyra"), "-o", str(work / "result")])
print(f"\nedit exit code {code}: the receipt records its own verdict; the statement below does not depend on it\n")

# 2. One P-256 key signs the model the OMS way and, below, the statement.
main(["keygen", "-o", str(work / "oms.key"), "--algorithm", "ecdsa-p256"])
result = work / "result"
oms("sign", "key", result, "--private_key", work / "oms.key", "--signature", work / "result.sig")
oms("sign", "key", base, "--private_key", work / "oms.key", "--signature", work / "base.sig")
print("OMS:", oms("verify", "key", result, "--signature", work / "result.sig", "--public_key", work / "oms.pub").stdout.strip())

# 3. The Neyra statement, joined to that OMS signature, signed with the same key.
change = work / "change"
main(["attest", str(base), str(result), "--patch", str(work / "edit.neyra"), "--receipt", str(work / "edit.receipt.json"),
      "--oms", str(work / "result.sig"), "--key", str(work / "oms.key"), "--attempts", "1", "-o", str(change)])
verify = ["verify", str(change / "change.statement.json"), "--key", str(work / "oms.pub"), "--base", str(base),
          "--result", str(result), "--patch", str(work / "edit.neyra")]
assert main(verify) == 0

# 4. Other people's tools.
print("\nstock tools:")
schemas = HERE.parent / "tests" / "fixtures" / "cyclonedx"
print(run(stock / "python", HERE / "attestation_stock.py", work, schemas).stdout)

# 5. Reshard the result: same tensors, three files and an index.
checkpoint = Checkpoint.open(str(result))
names = sorted(checkpoint.tensors)
again = work / "result-resharded"
again.mkdir(exist_ok=True)
weight_map = {}
for i in range(3):
    part = f"model-{i + 1:05d}-of-00003.safetensors"
    write_safetensors(again / part, {n: checkpoint.tensors[n].read() for n in names[i::3]}, {"format": "pt"}, "BF16")
    weight_map.update({n: part for n in names[i::3]})
(again / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
(again / "config.json").write_text((result / "config.json").read_text())
oms("sign", "key", again, "--private_key", work / "oms.key", "--signature", work / "resharded.sig")

statement = at.open_transformation(json.loads((change / "change.statement.json").read_text())).statement
oms_old = json.loads(base64.b64decode(json.loads((work / "result.sig").read_text())["dsseEnvelope"]["payload"]))
oms_new = json.loads(base64.b64decode(json.loads((work / "resharded.sig").read_text())["dsseEnvelope"]["payload"]))
stale = oms("verify", "key", again, "--signature", work / "result.sig", "--public_key", work / "oms.pub", check=False)
fresh = oms("verify", "key", again, "--signature", work / "resharded.sig", "--public_key", work / "oms.pub")
main(["attest", str(base), str(again), "--patch", str(work / "edit.neyra"), "--key", str(work / "oms.key"),
      "-o", str(work / "change-resharded")])
twin = at.open_transformation(json.loads((work / "change-resharded" / "change.statement.json").read_text())).statement
print("\nreshard (same tensors, 1 file -> 3 files + index):")
rows_ = [
    ("OMS subject digest", oms_old["subject"][0]["digest"]["sha256"][:16], oms_new["subject"][0]["digest"]["sha256"][:16]),
    ("OMS files", len(oms_old["predicate"]["resources"]), len(oms_new["predicate"]["resources"])),
    ("OMS verify, old signature on resharded dir", "-", "FAILED" if stale.returncode else "passed"),
    ("OMS verify, new signature", "-", "passed" if fresh.returncode == 0 else "FAILED"),
    ("Neyra weights root", root_of(tensor_table(Checkpoint.open(str(result))))[7:23],
     root_of(tensor_table(checkpoint.__class__.open(str(again))))[7:23]),
    ("Neyra statement, result-weights subject", statement["subject"][0]["digest"][at.WEIGHTS_KEY][:16],
     twin["subject"][0]["digest"][at.WEIGHTS_KEY][:16]),
    ("Neyra statement vs resharded dir", "ok", at.bind_model(statement, again)),
    ("Neyra statement for it, OMS subject", at.oms_root(statement["predicate"]["result"]["files"])[:16],
     at.oms_root(twin["predicate"]["result"]["files"])[:16]),
]
width = max(len(r[0]) for r in rows_)
print(f"{'':{width}}  {'original':18} resharded")
for label, before, after in rows_:
    print(f"{label:{width}}  {before!s:18} {after}")
print(f"\nwork directory {work}")
