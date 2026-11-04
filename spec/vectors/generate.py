"""Write the conformance vectors. Deterministic: rerunning gives identical files."""

import base64
import json
import shutil
import struct
from pathlib import Path

import numpy as np

from neyra.conformance import _run
from neyra.ledger import GENESIS, entry_digest
from neyra.patch import Patch, diff_tensor, write_safetensors
from neyra.protocol import digest, envelope, jcs, statement_for, tensor_table
from neyra.store import Checkpoint

ROOT = Path(__file__).parent
FIX = ROOT / "fixtures"
shutil.rmtree(FIX, ignore_errors=True)
FIX.mkdir()
rng = np.random.default_rng(1)
cases = []


def case(kind, id, expect=None, **fields):
    cases.append({"kind": kind, "id": id, **fields, "expect": expect})


for i, (text, code) in enumerate([
    ('{"b":1,"a":2}', None), ('{"\\ufb33":1,"\\ud83d\\ude00":2,"a":3}', None), ('[1, 2,  3]', None),
    ('"é\\n\\u0001\\""', None), ('{"a":true,"b":false,"c":null}', None), ('-7', None), ('9007199254740991', None),
    ('{"a":1,"a":2}', "bad_json"), ('1.5', "bad_json"), ('1e3', "bad_json"), ('9007199254740992', "bad_json"),
    ('NaN', "bad_json"), ('"\\ud800"', "bad_json"), ('{', "bad_json"), ('01', "bad_json"),
    ('-0', None), ('[' * 128 + ']' * 128, None), ('[' * 129 + ']' * 129, "bad_json"),
]):
    case("jcs", f"{i}", {"code": code} if code else None, input_hex=text.encode().hex())
case("jcs", "invalid-utf8", {"code": "bad_json"}, input_hex="22ff22")

for dom, data in (("payload", b""), ("patch", b"abc"), ("weights", b"\x00\xff"), ("ledger", "é".encode())):
    case("digest", dom, domain=dom, data_hex=data.hex())

for name, dtype, shape, raw in (
    ("bf16", "BF16", [2, 2], bytes(range(8))), ("same-bytes-f16", "F16", [2, 2], bytes(range(8))),
    ("same-bytes-reshaped", "BF16", [4], bytes(range(8))), ("scalar", "F32", [], bytes(4)),
    ("empty", "U8", [0], b""), ("bool", "BOOL", [3], b"\x00\x01\x00"),
):
    case("tensor", name, dtype=dtype, shape=shape, raw_hex=raw.hex())

a = rng.integers(0, 65535, (4, 6), dtype=np.uint16)
b = rng.integers(0, 65535, (8,), dtype=np.uint16)
c = rng.integers(0, 255, (5,), dtype=np.uint8)


def model(name, *shards):
    d = FIX / name
    d.mkdir()
    for i, tensors in enumerate(shards):
        write_safetensors(d / f"model-{i}.safetensors", tensors, {"format": "pt"}, "BF16")
    return d


base = model("base", {"w": a, "é.bias": b, "z": c})
model("base-resharded", {"z": c}, {"é.bias": b, "w": a})
edited = a.copy()
edited[1, 2] ^= 7
edited[3, 5] ^= 1
model("other", {"w": a, "é.bias": b + 1, "z": c})
case("root", "one-shard", model="fixtures/base")
case("root", "two-shards-same-root", model="fixtures/base-resharded")
case("root", "different", model="fixtures/other")
model("duplicate-name", {"w": a}, {"w": a})
(FIX / "empty").mkdir()
(FIX / "empty" / ".keep").write_text("")
case("root", "same-tensor-in-two-files", {"code": "bad_safetensors"}, model="fixtures/duplicate-name")
case("root", "no-weight-files", {"code": "bad_safetensors"}, model="fixtures/empty")

ck = Checkpoint.open(str(base))
good = Patch([diff_tensor("w", "BF16", a, edited)], {"note": "fixture"}).bind(tensor_table(ck))
good.save(FIX / "good.neyra")
case("patch", "valid-with-base", {"code": "ok"}, file="fixtures/good.neyra", base="fixtures/base")
case("patch", "valid-no-base", {"code": "ok"}, file="fixtures/good.neyra")
case("patch", "wrong-base", {"code": "base_mismatch"}, file="fixtures/good.neyra", base="fixtures/other")


def tampered(name, change, payload=None, expect=None, with_base=False):
    doc = json.loads(jcs(good.manifest()))
    change(doc)
    e = good.edits[0]
    tensors = {"0.idx": e.idx.astype("<i8"), "0.val": e.val, **(payload or {})}
    raw = jcs(doc).decode()
    write_safetensors(FIX / name, tensors, {"neyra.manifest": raw})
    case("patch", name.removesuffix(".neyra"), {"code": expect}, file=f"fixtures/{name}",
         **({"base": "fixtures/base"} if with_base else {}))


def set_key(k, v):
    return lambda d: d.update({k: v})


def fix_payload(idx=None, val=None):
    def change(d):
        t = d["tensors"][0]
        if idx is not None:
            t["idx_digest"] = digest("payload", idx.tobytes())
        if val is not None:
            t["val_digest"] = digest("payload", val.tobytes())
    return change


tampered("bad-type.neyra", set_key("type", "neyra.patch/2"), expect="bad_type")
tampered("unknown-member.neyra", set_key("surprise", 1), expect="unknown_member")
tampered("x-member-ok.neyra", set_key("x-note", "fine"), expect="ok")
tampered("unknown-extension.neyra", set_key("requires", ["quantum/1"]), expect="unknown_extension")
tampered("wrong-val-digest.neyra", lambda d: d["tensors"][0].update({"val_digest": "sha256:" + "0" * 64}), expect="payload_mismatch")
oob = np.array([7, 999], dtype="<i8")
tampered("index-out-of-range.neyra", fix_payload(idx=oob), {"0.idx": oob}, "bad_index")
back = np.array([9, 3], dtype="<i8")
tampered("index-unsorted.neyra", fix_payload(idx=back), {"0.idx": back}, "bad_index")
dup = np.array([3, 3], dtype="<i8")
tampered("index-duplicate.neyra", fix_payload(idx=dup), {"0.idx": dup}, "bad_index")
tampered("missing-result-root.neyra", lambda d: d.pop("result_root"), expect="bad_manifest")
tampered("malformed-digest.neyra", set_key("base_root", "sha256:XYZ"), expect="bad_manifest")
tampered("requires-not-a-list.neyra", set_key("requires", "low-rank/1"), expect="bad_manifest")
tampered("tensor-member-missing.neyra", lambda d: d["tensors"][0].pop("val_digest"), expect="bad_manifest")
wide_val = np.array([1, 2], dtype=np.uint32)
tampered("val-wrong-dtype.neyra", fix_payload(val=wide_val), {"0.val": wide_val}, "bad_manifest")
short_val = np.array([1], dtype=np.uint16)
tampered("length-mismatch.neyra", fix_payload(val=short_val), {"0.val": short_val}, "bad_manifest")
tampered("wrong-base-digest.neyra", lambda d: d["tensors"][0].update({"base_digest": "sha256:" + "3" * 64}),
         expect="base_mismatch", with_base=True)
tampered("wrong-result-digest.neyra", lambda d: d["tensors"][0].update({"result_digest": "sha256:" + "1" * 64}),
         expect="result_mismatch", with_base=True)
tampered("wrong-result-root.neyra", set_key("result_root", "sha256:" + "2" * 64), expect="result_mismatch", with_base=True)

raw_manifest = jcs(good.manifest()).decode()
e = good.edits[0]
write_safetensors(FIX / "not-canonical.neyra", {"0.idx": e.idx.astype("<i8"), "0.val": e.val},
                  {"neyra.manifest": raw_manifest.replace('{"base_root"', '{ "base_root"')})
case("patch", "not-canonical", {"code": "not_canonical"}, file="fixtures/not-canonical.neyra")
write_safetensors(FIX / "no-manifest.neyra", {"0.idx": e.idx.astype("<i8")}, {"other": "x"})
case("patch", "no-manifest", {"code": "bad_type"}, file="fixtures/no-manifest.neyra")

blob = (FIX / "good.neyra").read_bytes()
(FIX / "trailing-bytes.neyra").write_bytes(blob + b"x")
case("patch", "trailing-bytes", {"code": "bad_safetensors"}, file="fixtures/trailing-bytes.neyra")
(FIX / "short.neyra").write_bytes(b"\x01\x02")
case("patch", "too-short", {"code": "bad_safetensors"}, file="fixtures/short.neyra")
(FIX / "huge-header.neyra").write_bytes(struct.pack("<Q", 2**40) + b"{}")
case("patch", "huge-header", {"code": "bad_safetensors"}, file="fixtures/huge-header.neyra")
h = b'{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}'
(FIX / "overlap.neyra").write_bytes(struct.pack("<Q", len(h)) + h + bytes(6))
case("patch", "overlapping-offsets", {"code": "bad_safetensors"}, file="fixtures/overlap.neyra")

entries = []
head = GENESIS
for i, (kind, body) in enumerate((("open", {"model": "é", "n": 3}), ("sample", {"metric": "0.500000"}), ("submit", {"verdict": "PASS"}))):
    entry = {"seq": i, "prev": head, "time": f"2026-11-04T12:0{i}:00Z", "kind": kind, "body": body}
    entries.append(entry)
    head = entry_digest(entry)
lines = [jcs(x) for x in entries]


def ledger(name, rows, tail=b"\n", expect=None):
    (FIX / name).write_bytes(b"\n".join(rows) + tail)
    case("ledger", name.removesuffix(".jsonl"), expect, file=f"fixtures/{name}")


ledger("ledger-good.jsonl", lines)
ledger("ledger-edited.jsonl", [lines[0], lines[1].replace(b"0.500000", b"0.900000"), lines[2]], expect={"code": "bad_chain"})
ledger("ledger-dropped.jsonl", [lines[0], lines[2]], expect={"code": "bad_chain"})
ledger("ledger-swapped.jsonl", [lines[1], lines[0], lines[2]], expect={"code": "bad_chain"})
ledger("ledger-torn.jsonl", lines, tail=b"\n{\"seq\":3", expect={"code": "bad_chain"})
ledger("ledger-no-final-newline.jsonl", lines, tail=b"", expect={"code": "bad_chain"})
ledger("ledger-bad-time.jsonl", [lines[0].replace(b"2026-11-04T12:00:00Z", b"yesterday"), lines[1], lines[2]], expect={"code": "bad_chain"})
ledger("ledger-not-json.jsonl", [b"not json", lines[1], lines[2]], expect={"code": "bad_chain"})
ledger("ledger-not-canonical.jsonl", [lines[0].replace(b'{"body"', b'{ "body"'), lines[1], lines[2]], expect={"code": "bad_chain"})

secret = bytes(range(32))
other_secret = bytes(range(1, 33))
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def public_of(raw):
    return Ed25519PrivateKey.from_private_bytes(raw).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


claim = {"type": "neyra.claim/1", "base_root": good.base_root, "result_root": good.result_root, "patch_id": good.id,
         "task": {"id": "t", "holdout_commit": digest("holdout", bytes(32) + b"[]")},
         "eval_env": {"runtime": "fixture"}, "gates": {"effect": True}, "limits": {"passes_used": 1},
         "statements": [{"metric": "effect", "estimate": "0.100000", "ci_low": "0.010000", "ci_high": "0.200000",
                         "n": 100, "method": "paired bootstrap"}], "verdict": "PASS"}
statement = statement_for(claim, good.id, head, 3)


def receipt(name, doc, expect, public=None):
    (FIX / name).write_text(json.dumps(doc))
    case("receipt", name.removesuffix(".json"), {"code": expect}, file=f"fixtures/{name}",
         **({"public_hex": public} if public else {}))


signed = envelope(statement, secret)
pub = public_of(secret)
receipt("receipt-signed.json", signed, "ok", pub)
receipt("receipt-no-key-given.json", signed, "ok")
receipt("receipt-unsigned.json", envelope(statement), "ok")
receipt("receipt-unsigned-but-key-given.json", envelope(statement), "bad_signature", pub)
receipt("receipt-wrong-key.json", signed, "bad_signature", public_of(other_secret))
flipped = json.loads(jcs(statement))
flipped["predicate"]["claim"]["verdict"] = "FAIL"
receipt("receipt-tampered-claim.json", {**signed, "payload": base64.b64encode(jcs(flipped)).decode()}, "bad_signature", pub)
wrong = json.loads(jcs(statement))
wrong["subject"][0]["digest"]["sha256"] = "f" * 64
receipt("receipt-wrong-subject.json", envelope(wrong, secret), "subject_mismatch", pub)
loose = base64.b64encode(json.dumps(statement).encode()).decode()
receipt("receipt-not-canonical.json", {**signed, "payload": loose}, "not_canonical", pub)
receipt("receipt-bad-type.json", {**signed, "payloadType": "text/plain"}, "bad_type", pub)
stype = json.loads(jcs(statement))
stype["predicateType"] = "https://example.com/other"
receipt("receipt-bad-predicate.json", envelope(stype, secret), "bad_type", pub)

missing = json.loads(jcs(statement))
del missing["predicate"]["claim"]["verdict"]
receipt("receipt-claim-incomplete.json", envelope(missing, secret), "bad_claim", pub)
noled = json.loads(jcs(statement))
del noled["predicate"]["ledger_head"]
receipt("receipt-no-ledger-head.json", envelope(noled, secret), "bad_claim", pub)
extra_subject = json.loads(jcs(statement))
extra_subject["subject"].append({"name": "other", "digest": {"sha256": "0" * 64}})
receipt("receipt-three-subjects.json", envelope(extra_subject, secret), "subject_mismatch", pub)
reversed_subject = json.loads(jcs(statement))
reversed_subject["subject"].reverse()
receipt("receipt-subjects-reversed-ok.json", envelope(reversed_subject, secret), "ok", pub)
receipt("receipt-bad-sig-base64.json", {**signed, "signatures": [{**signed["signatures"][0], "sig": "!!!"}]}, "bad_signature", pub)
two = envelope(statement, other_secret)["signatures"] + signed["signatures"]
receipt("receipt-two-signatures.json", {**signed, "signatures": two}, "ok", pub)

for c_ in cases:
    if c_["expect"] is None or (c_["kind"] == "ledger" and c_["expect"] is None) or c_["kind"] == "ledger" and c_["expect"] == {"code": "ok"}:
        c_["expect"] = _run(ROOT, c_)

(ROOT / "cases.json").write_text(json.dumps(cases, indent=1, ensure_ascii=True) + "\n")
print(len(cases), "cases")
