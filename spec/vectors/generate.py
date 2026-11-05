"""Write the conformance vectors. Deterministic: rerunning gives identical files."""

import base64
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np

from neyra.conformance import _run
from neyra.ledger import GENESIS, entry_digest
from neyra.patch import Patch, diff_tensor, write_safetensors
from neyra.protocol import bits_digest, digest, envelope, jcs, statement_for, tensor_table
from neyra.rows import build_index, fetch, prove
from neyra.store import Checkpoint, LocalBlob, parse_header

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
    ('-0', None), ('{"a":' * 127 + '1' + '}' * 127, None), ('[' * 128 + ']' * 128, None), ('[' * 129 + ']' * 129, "bad_json"),
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
tampered("unknown-tensor-member.neyra", lambda d: d["tensors"][0].update({"surprise": 1}), expect="unknown_member")
tampered("x-tensor-member-ok.neyra", lambda d: d["tensors"][0].update({"x-note": "fine"}), expect="ok")
tampered("unknown-member-beats-bad-manifest.neyra", lambda d: (d.update({"surprise": 1}), d.pop("result_root")), expect="unknown_member")
tampered("bad-manifest-beats-unknown-extension.neyra", lambda d: (d.update({"requires": ["quantum/1"]}), d.pop("result_root")), expect="bad_manifest")
tampered("extra-payload-tensor.neyra", lambda d: None, {"9.extra": np.zeros(2, dtype=np.uint8)}, "bad_manifest")
tampered("repeated-tensor-name.neyra", lambda d: d["tensors"].append(dict(d["tensors"][0])), expect="bad_manifest")
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
(FIX / "ledger-empty.jsonl").write_bytes(b"")
case("ledger", "ledger-empty", file="fixtures/ledger-empty.jsonl")
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

wrong_alg = json.loads(jcs(statement))
wrong_alg["predicate"]["digest_algorithm"] = "other/v9"
receipt("receipt-wrong-digest-algorithm.json", envelope(wrong_alg, secret), "bad_claim", pub)
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



def binding(id, expect, receipt_file="fixtures/receipt-signed.json", ledger_file="fixtures/ledger-good.jsonl",
            patch_file="fixtures/good.neyra"):
    case("binding", id, {"code": expect}, receipt=receipt_file, ledger=ledger_file, patch=patch_file)


binding("complete", "ok")
head2 = entry_digest(entries[1])
receipt("receipt-head-mid-ledger.json", envelope(statement_for(claim, good.id, head2, 2), secret), "ok", pub)
binding("head-mid-ledger-is-fine", "ok", "fixtures/receipt-head-mid-ledger.json")
receipt("receipt-wrong-length.json", envelope(statement_for(claim, good.id, head, 2), secret), "ok", pub)
binding("wrong-ledger-len", "binding_mismatch", "fixtures/receipt-wrong-length.json")
receipt("receipt-unknown-head.json", envelope(statement_for(claim, good.id, "sha256:" + "4" * 64, 3), secret), "ok", pub)
binding("head-not-in-ledger", "binding_mismatch", "fixtures/receipt-unknown-head.json")
other_patch = Patch([diff_tensor("w", "BF16", a, a ^ np.uint16(1))]).bind(tensor_table(ck))
other_patch.save(FIX / "other-patch.neyra")
binding("different-patch", "binding_mismatch", patch_file="fixtures/other-patch.neyra")
binding("ledger-fails-alone", "bad_chain", ledger_file="fixtures/ledger-edited.jsonl")

# Annex A: low rank patches. The float arithmetic is the part two languages can disagree on.
from neyra.numerics import from_float
from neyra.patch import TensorEdit

lr_rng = np.random.default_rng(11)


def floats(dtype, *shape, scale=1.0):
    return from_float((lr_rng.normal(size=shape) * scale).astype(np.float32), dtype)


def explicit_edit(name, dtype, base, left, right):
    edit = TensorEdit(name, dtype, base.shape, bits_digest(dtype, base), "", np.zeros(0, np.int64),
                      np.zeros(0, base.dtype), np.array(left, np.float64), np.array(right, np.float64))
    edit.result_digest = bits_digest(dtype, edit.apply(base, verify=False))
    return edit


def low_rank_edit(name, dtype, base, rank, scale, corrections=0):
    rows, cols = base.shape
    left, right = lr_rng.normal(size=(rows, rank)) * scale, lr_rng.normal(size=(rank, cols))
    edit = TensorEdit(name, dtype, base.shape, bits_digest(dtype, base), "", np.zeros(0, np.int64),
                      np.zeros(0, base.dtype), left, right)
    built = edit.apply(base, verify=False).reshape(-1)
    if corrections:
        edit.idx = np.sort(lr_rng.choice(built.size, corrections, replace=False)).astype(np.int64)
        edit.val = (built[edit.idx] ^ base.dtype.type(1)).astype(base.dtype)
        built = built.copy()
        built[edit.idx] = edit.val
    edit.result_digest = bits_digest(dtype, built.reshape(base.shape))
    return edit


tiny = floats("F16", 6, 5, scale=3e-4)
tiny[0, 0] = from_float(np.float32([6.0e4]), "F16")[0]
lr_base = FIX / "lowrank-base"
lr_base.mkdir()
bf_base, f32_base = floats("BF16", 7, 9), lr_rng.normal(size=(4, 6)).astype(np.float32)
bf_ties = from_float(np.float32([[1.0, 1.0]]), "BF16")
write_safetensors(lr_base / "model-0.safetensors", {"bf": bf_base, "bf_ties": bf_ties}, {"format": "pt"}, "BF16")
wide = from_float(np.clip(lr_rng.normal(size=(3, 4)) * 4e4, -6e4, 6e4).astype(np.float32), "F16")
half_ties = from_float(np.float32([[1.0, 1.0, 0.0, 0.0]]), "F16")
write_safetensors(lr_base / "model-1.safetensors", {"half": tiny, "wide": wide, "half_ties": half_ties}, {"format": "pt"}, "F16")
ones = np.ones((1, 1), np.float32)
write_safetensors(lr_base / "model-2.safetensors", {"single": f32_base, "plain": b, "order": ones, "tie": ones.copy()},
                  {"format": "pt"})
lr_ck = Checkpoint.open(str(lr_base))
lr_table = tensor_table(lr_ck)
lowrank = Patch([low_rank_edit("bf", "BF16", bf_base, 2, 0.05, corrections=3),
                 low_rank_edit("half", "F16", tiny, 1, 4e-4),
                 low_rank_edit("single", "F32", f32_base.view(np.uint32), 3, 0.02),
                 low_rank_edit("wide", "F16", wide, 1, 9e4),
                 # exact midpoints between two neighbours: the even one wins, up or down
                 explicit_edit("bf_ties", "BF16", bf_ties, [[1.0]], [[2.0**-8, 3 * 2.0**-8]]),
                 explicit_edit("half_ties", "F16", half_ties, [[1.0]], [[2.0**-11, 3 * 2.0**-11, 2.0**-25, 3 * 2.0**-25]]),
                 # 2^53 + 1 does not exist in float64: added first the 1 is lost, added last it survives
                 explicit_edit("order", "F32", ones.view(np.uint32), [[2.0**53, -(2.0**53)]], [[1.0], [1.0]]),
                 # a product that lands on a rounding midpoint only if it is rounded before the add
                 explicit_edit("tie", "F32", ones.view(np.uint32), [[float.fromhex("0x1.e96b96aac6da2p-24")]],
                               [[float.fromhex("0x1.91b752265b1f5p+0")]])]).bind(lr_table)
lowrank.save(FIX / "lowrank.neyra")
case("patch", "lowrank-three-dtypes-with-base", {"code": "ok"}, file="fixtures/lowrank.neyra", base="fixtures/lowrank-base")
case("patch", "lowrank-no-base", {"code": "ok"}, file="fixtures/lowrank.neyra")
case("patch", "lowrank-on-a-different-base", {"code": "base_mismatch"}, file="fixtures/lowrank.neyra", base="fixtures/base")


def lowrank_variant(name, expect, manifest=None, tensors=None, with_base=True):
    refs, meta = parse_header(LocalBlob(FIX / "lowrank.neyra"))
    doc = json.loads(meta["neyra.manifest"])
    kinds = {"I64": np.int64, "F64": np.float64}
    payload = {k: r.read().view(kinds[r.dtype]) if r.dtype in kinds else r.read() for k, r in refs.items()}
    if manifest:
        manifest(doc)
    if tensors:
        tensors(payload)
    write_safetensors(FIX / name, payload, {"neyra.manifest": jcs(doc).decode()})
    case("patch", name.removesuffix(".neyra"), {"code": expect}, file=f"fixtures/{name}",
         **({"base": "fixtures/lowrank-base"} if with_base else {}))


SINGLE = sorted(e.name.encode("utf-16-be") for e in lowrank.edits).index("single".encode("utf-16-be"))


def entry(i):
    return lambda d: d["tensors"][i]


lowrank_variant("lowrank-left-digest-wrong.neyra", "payload_mismatch",
                lambda d: entry(0)(d).update(left_digest="sha256:" + "1" * 64), with_base=False)
lowrank_variant("lowrank-right-digest-wrong.neyra", "payload_mismatch",
                lambda d: entry(1)(d).update(right_digest="sha256:" + "1" * 64), with_base=False)
lowrank_variant("lowrank-result-digest-wrong.neyra", "result_mismatch",
                lambda d: entry(SINGLE)(d).update(result_digest="sha256:" + "2" * 64))
lowrank_variant("lowrank-left-without-right.neyra", "bad_manifest", lambda d: entry(0)(d).pop("right_digest"), with_base=False)
lowrank_variant("lowrank-extension-not-required.neyra", "bad_manifest", lambda d: d.update(requires=[]), with_base=False)
lowrank_variant("lowrank-other-extension-too.neyra", "unknown_extension",
                lambda d: d.update(requires=["low-rank/1", "quantum/1"]), with_base=False)
lowrank_variant("lowrank-on-an-integer-tensor.neyra", "bad_manifest", lambda d: entry(0)(d).update(dtype="I8"), with_base=False)
lowrank_variant("lowrank-on-a-vector.neyra", "bad_manifest", lambda d: entry(0)(d).update(shape=[63]), with_base=False)
lowrank_variant("lowrank-left-missing.neyra", "bad_manifest", tensors=lambda p: p.pop("0.left"), with_base=False)
lowrank_variant("lowrank-left-is-float32.neyra", "bad_manifest",
                tensors=lambda p: p.update({"0.left": p["0.left"].astype(np.float32)}), with_base=False)
lowrank_variant("lowrank-inner-sizes-differ.neyra", "bad_manifest",
                tensors=lambda p: p.update({"0.left": p["0.left"][:, :1].copy()}), with_base=False)
lowrank_variant("lowrank-rank-zero.neyra", "bad_manifest",
                tensors=lambda p: p.update({"1.left": p["1.left"][:, :0].copy(), "1.right": p["1.right"][:0].copy()}),
                with_base=False)
lowrank_variant("lowrank-factor-rows-wrong.neyra", "bad_manifest",
                tensors=lambda p: p.update({f"{SINGLE}.left": p[f"{SINGLE}.left"][:3].copy()}), with_base=False)

# Annex C: transformation statements, status records and P-256 signatures.
import copy

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from neyra import attest as at
from neyra.protocol import key_id, pae, root_of

# Published test keys, like the Ed25519 seed above. Nothing real is signed with them.
P256 = ec.derive_private_key(int.from_bytes(hashlib.sha256(b"neyra conformance vectors, p256").digest(), "big"), ec.SECP256R1())
P256_OTHER = ec.derive_private_key(int.from_bytes(hashlib.sha256(b"neyra conformance vectors, other").digest(), "big"), ec.SECP256R1())
P384 = ec.derive_private_key(int.from_bytes(hashlib.sha256(b"neyra conformance vectors, p384").digest(), "big"), ec.SECP384R1())


def der(private):
    return private.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)


def p256_envelope(statement, private=P256, payload=None):
    payload = payload if payload is not None else jcs(statement)
    sig = private.sign(pae("application/vnd.in-toto+json", payload), ec.ECDSA(hashes.SHA256(), deterministic_signing=True))
    return {"payloadType": "application/vnd.in-toto+json", "payload": base64.b64encode(payload).decode(),
            "signatures": [{"keyid": key_id(der(private)), "sig": base64.b64encode(sig).decode()}]}


receipt("receipt-signed-p256.json", p256_envelope(statement), "ok", der(P256).hex())
receipt("receipt-p256-no-key-given.json", p256_envelope(statement), "ok")
receipt("receipt-p256-wrong-key.json", p256_envelope(statement), "bad_signature", der(P256_OTHER).hex())
receipt("receipt-p256-key-for-ed25519-signature.json", signed, "bad_signature", der(P256).hex())
receipt("receipt-ed25519-key-for-p256-signature.json", p256_envelope(statement), "bad_signature", pub)
receipt("receipt-p256-wrong-curve-key.json", p256_envelope(statement), "bad_signature", der(P384).hex())
receipt("receipt-p256-tampered-claim.json",
        {**p256_envelope(statement), "payload": base64.b64encode(jcs(flipped)).decode()}, "bad_signature", der(P256).hex())

RESULT_BITS = a.copy()
RESULT_BITS[1, 2] ^= 7
RESULT_BITS[3, 5] ^= 1


def result_model(name, *shards, config='{"hidden_size": 4}', extra=()):
    d = model(name, *shards)
    (d / "config.json").write_text(config)
    (d / "docs").mkdir()
    (d / "docs" / "README.md").write_text("fixture\n")
    (d / "docs.txt").write_text("sorts after docs/ by component\n")
    for file in extra:
        (d / file).write_text("{}")
    return d


whole = {"w": RESULT_BITS, "é.bias": b, "z": c}
result_dir = result_model("result-model", whole)
(result_dir / ".gitignore").write_text("*.tmp\n")
result_model("result-model-resharded", {"z": c}, {"é.bias": b, "w": RESULT_BITS})
result_model("result-model-tampered-config", whole, config='{"hidden_size": 5}')
signed_dir = result_model("result-model-signed", whole, extra=("model.sig",))

result_ck = Checkpoint.open(str(result_dir))
before, after = tensor_table(ck), tensor_table(result_ck)
TOTAL = sum(r.nbytes for model_ck in (ck, result_ck) for r in model_ck.tensors.values())
GOOD_T = at.transformation(
    kind="edit", grade="exact", meaning="authored", base_root=good.base_root, result_root=good.result_root,
    base_files=at.file_digests(FIX / "base"), result_files=at.file_digests(result_dir), result_name="result-model",
    recipe={"patch_id": good.id}, extent=at.measure(ck, result_ck, before, after),
    coverage={"bytes_read": TOTAL, "bytes_total": TOTAL}, time="2026-11-05T12:00:00Z")
GOOD_T["predicate"]["tool"]["version"] = "0.1.0"
assert good.result_root == root_of(after)
TIME2 = "2026-11-06T09:00:00Z"


def variant(change):
    s = copy.deepcopy(GOOD_T)
    change(s["predicate"])
    s["subject"] = at.subjects_of(s["predicate"])
    return s


def keyed(statement, key):
    return {"ed25519": lambda: envelope(statement, secret), "p256": lambda: p256_envelope(statement),
            "none": lambda: envelope(statement)}[key]()


def public_for(key):
    return {"ed25519": pub, "p256": der(P256).hex(), "none": None}[key]


def transform(name, statement, expect, key="ed25519", public=True, doc=None):
    (FIX / name).write_text(json.dumps(doc or keyed(statement, key)))
    case("transform", name.removesuffix(".json"), {"code": expect}, file=f"fixtures/{name}",
         **({"public_hex": public_for(key)} if public and public_for(key) else {}))


def payload_of(statement, **swap):
    return {**envelope(statement, secret), **swap}


transform("transform-signed-ed25519.json", GOOD_T, "ok")
transform("transform-signed-p256.json", GOOD_T, "ok", "p256")
transform("transform-no-key-given.json", GOOD_T, "ok", public=False)
transform("transform-unsigned.json", GOOD_T, "ok", "none", public=False)
(FIX / "transform-unsigned-but-key-given.json").write_text(json.dumps(envelope(GOOD_T)))
case("transform", "transform-unsigned-but-key-given", {"code": "bad_signature"},
     file="fixtures/transform-unsigned-but-key-given.json", public_hex=pub)
(FIX / "transform-wrong-key.json").write_text(json.dumps(p256_envelope(GOOD_T, P256_OTHER)))
case("transform", "transform-wrong-key", {"code": "bad_signature"}, file="fixtures/transform-wrong-key.json",
     public_hex=der(P256).hex())
cut = copy.deepcopy(GOOD_T)
cut["predicate"]["extent"]["changed"] += 1
transform("transform-tampered-after-signing.json", GOOD_T, "bad_signature",
          doc=payload_of(GOOD_T, payload=base64.b64encode(jcs(cut)).decode()))
transform("transform-not-canonical.json", GOOD_T, "not_canonical",
          doc=payload_of(GOOD_T, payload=base64.b64encode(json.dumps(GOOD_T, indent=1).encode()).decode()))
transform("transform-bad-payload-type.json", GOOD_T, "bad_type", doc=payload_of(GOOD_T, payloadType="text/plain"))
transform("transform-duplicate-key.json", GOOD_T, "bad_json",
          doc=payload_of(GOOD_T, payload=base64.b64encode(b'{"_type":"x","_type":"y"}').decode()))
wrong_type = copy.deepcopy(GOOD_T)
wrong_type["predicateType"] = "https://neyra.run/receipt/v1"
transform("transform-wrong-predicate-type.json", wrong_type, "bad_type")
transform("transform-kind-outside-vocabulary.json", variant(lambda p: p.update(kind="rebase")), "bad_transform")
transform("transform-grade-outside-vocabulary.json", variant(lambda p: p.update(grade="certain")), "bad_transform")
transform("transform-meaning-outside-vocabulary.json", variant(lambda p: p.update(meaning="blessed")), "bad_transform")
transform("transform-not-measured-incomplete.json", variant(lambda p: p.update(not_measured=["behaviour", "training data"])), "bad_transform")
transform("transform-not-measured-extra-ok.json", variant(lambda p: p["not_measured"].append("latency")), "ok")
transform("transform-exact-without-patch-id.json", variant(lambda p: p.update(recipe={})), "bad_transform")
transform("transform-unknown-kind-cannot-be-exact.json", variant(lambda p: p.update(kind="unknown")), "bad_transform")
transform("transform-declared-without-extent-ok.json",
          variant(lambda p: p.update(grade="declared", kind="quantize", extent=None, recipe={"scheme": "q4"},
                                     basis=at.basis_for("declared"), coverage={"bytes_read": 0, "bytes_total": TOTAL})), "ok")
transform("transform-observed-without-extent.json",
          variant(lambda p: p.update(grade="observed", extent=None, basis=at.basis_for("observed"))), "bad_transform")
transform("transform-declared-with-the-basis-of-an-exact-one.json",
          variant(lambda p: p.update(grade="declared", kind="quantize", extent=None, recipe={},
                                     coverage={"bytes_read": 0, "bytes_total": TOTAL})), "bad_transform")
transform("transform-basis-moves-a-field.json", variant(lambda p: p["basis"]["asserted"].remove("attempts")), "bad_transform")
transform("transform-sampled-with-the-basis-of-an-exact-one.json",
          variant(lambda p: p.update(grade="sampled", kind="unknown", recipe={},
                                     coverage={"bytes_read": TOTAL - 8, "bytes_total": TOTAL})), "bad_transform")
transform("transform-sampled-ok.json",
          variant(lambda p: p.update(grade="sampled", kind="unknown", recipe={}, basis=at.basis_for("sampled"),
                                     coverage={"bytes_read": TOTAL - 8, "bytes_total": TOTAL})), "ok")
transform("transform-exact-but-partly-read.json",
          variant(lambda p: p["coverage"].update(bytes_read=TOTAL - 8)), "bad_transform")
transform("transform-sampled-but-fully-read.json",
          variant(lambda p: p.update(grade="sampled", kind="unknown", recipe={}, basis=at.basis_for("sampled"))), "bad_transform")
transform("transform-declared-but-bytes-read.json",
          variant(lambda p: p.update(grade="declared", kind="quantize", extent=None, recipe={},
                                     basis=at.basis_for("declared"))), "bad_transform")
transform("transform-coverage-missing.json", variant(lambda p: p.pop("coverage")), "bad_transform")
transform("transform-attempts-counted-ok.json", variant(lambda p: p.update(attempts=3)), "ok")
transform("transform-attempts-zero.json", variant(lambda p: p.update(attempts=0)), "bad_transform")
transform("transform-attempts-a-string.json", variant(lambda p: p.update(attempts="3")), "bad_transform")
transform("transform-extent-not-integers.json", variant(lambda p: p["extent"].update(changed="1")), "bad_transform")
transform("transform-extent-member-missing.json", variant(lambda p: p["extent"].pop("removed")), "bad_transform")
transform("transform-rederive-empty.json", variant(lambda p: p.update(rederive=[])), "bad_transform")
transform("transform-time-malformed.json", variant(lambda p: p.update(time="yesterday")), "bad_transform")
transform("transform-unknown-member.json", variant(lambda p: p.update(surprise=1)), "bad_transform")
transform("transform-x-member-ok.json", variant(lambda p: p.update({"x-note": "kept"})), "ok")
transform("transform-x-member-inside-result-refused.json", variant(lambda p: p["result"].update({"x-note": "no"})), "bad_transform")
transform("transform-member-missing.json", variant(lambda p: p.pop("evidence")), "bad_transform")
transform("transform-evidence-bad-digest.json", variant(lambda p: p.update(evidence=[{"role": "patch", "digest": "nope"}])), "bad_transform")
transform("transform-tool-incomplete.json", variant(lambda p: p["tool"].pop("version")), "bad_transform")
transform("transform-root-not-a-digest.json", variant(lambda p: p["base"].update(weights_root="sha256:XYZ")), "bad_transform")
transform("transform-ignore-not-a-list.json", variant(lambda p: p["result"].update(ignore="model.sig")), "bad_transform")
transform("transform-files-sorted-as-strings.json",
          variant(lambda p: p["result"].update(files=sorted(p["result"]["files"], key=lambda f: f["path"]))), "bad_transform")
transform("transform-files-repeat-a-path.json", variant(lambda p: p["result"]["files"].insert(1, dict(p["result"]["files"][0]))), "bad_transform")
transform("transform-file-path-escapes.json", variant(lambda p: p["result"]["files"].insert(0, {"path": "../x", "sha256": "0" * 64})), "bad_transform")
transform("transform-file-digest-uppercase.json", variant(lambda p: p["result"]["files"][0].update(sha256="A" * 64)), "bad_transform")
transform("transform-no-files-ok.json", variant(lambda p: p["result"].update(files=[], name="")), "ok")
transform("transform-files-without-name.json", variant(lambda p: p["result"].update(name="")), "bad_transform")
no_file_subject = copy.deepcopy(GOOD_T)
no_file_subject["subject"] = [s for s in no_file_subject["subject"] if s["name"] != "docs.txt"]
transform("transform-subject-missing-a-file.json", no_file_subject, "subject_mismatch")
extra_subject = copy.deepcopy(GOOD_T)
extra_subject["subject"].append({"name": "extra", "digest": {"sha256": "1" * 64}})
transform("transform-subject-extra.json", extra_subject, "subject_mismatch")
plain_root = copy.deepcopy(GOOD_T)
plain_root["subject"][0]["digest"] = {"sha256": plain_root["subject"][0]["digest"][at.WEIGHTS_KEY]}
transform("transform-weights-subject-under-sha256.json", plain_root, "subject_mismatch")
oms_less = copy.deepcopy(GOOD_T)
oms_less["subject"] = [s for s in oms_less["subject"] if s["name"] != "result-model"]
transform("transform-subject-missing-oms-root.json", oms_less, "subject_mismatch")
reordered = copy.deepcopy(GOOD_T)
reordered["subject"].reverse()
transform("transform-subjects-in-any-order-ok.json", reordered, "ok")

# Binding to a patch
BROKEN = variant(lambda p: p.update(kind="rebase"))
for name, st in (("good", GOOD_T), ("other-patch-id", variant(lambda p: p["recipe"].update(patch_id=other_patch.id))),
                 ("other-base", variant(lambda p: p["base"].update(weights_root="sha256:" + "7" * 64))),
                 ("other-result", variant(lambda p: p["result"].update(weights_root="sha256:" + "8" * 64))),
                 ("broken", BROKEN)):
    (FIX / f"bind-{name}.json").write_text(json.dumps(envelope(st)))


def tbind(id, expect, statement, patch="fixtures/good.neyra"):
    case("transformbinding", id, {"code": expect}, statement=f"fixtures/bind-{statement}.json", patch=patch)


tbind("complete", "ok", "good")
tbind("patch-id-differs", "binding_mismatch", "other-patch-id")
tbind("base-root-differs", "binding_mismatch", "other-base")
tbind("result-root-differs", "binding_mismatch", "other-result")
tbind("a-different-patch-given", "binding_mismatch", "good", "fixtures/other-patch.neyra")
tbind("statement-fails-alone", "bad_transform", "broken")
tbind("patch-fails-alone", "bad_type", "good", "fixtures/bad-type.neyra")

# Binding to a model directory
for name, st in (("files", GOOD_T),
                 ("ignoring-signature", variant(lambda p: p["result"].update(
                     files=at.file_digests(signed_dir, ("model.sig",)), ignore=["model.sig"]))),
                 ("no-files", variant(lambda p: p["result"].update(files=[], name=""))), ("broken", BROKEN)):
    (FIX / f"files-{name}.json").write_text(json.dumps(envelope(st)))


def tfiles(id, expect, statement, model):
    case("transformfiles", id, {"code": expect}, statement=f"fixtures/files-{statement}.json", model=f"fixtures/{model}")


tfiles("complete-with-a-gitignore-that-is-not-listed", "ok", "files", "result-model")
tfiles("resharded-keeps-the-root-and-loses-the-files", "files_mismatch", "files", "result-model-resharded")
tfiles("resharded-without-files-is-fine", "ok", "no-files", "result-model-resharded")
tfiles("config-edited", "files_mismatch", "files", "result-model-tampered-config")
tfiles("other-weights", "result_mismatch", "files", "base")
tfiles("signature-in-directory-named-as-ignored", "ok", "ignoring-signature", "result-model-signed")
tfiles("signature-in-directory-not-ignored", "files_mismatch", "files", "result-model-signed")
tfiles("no-weights", "bad_safetensors", "files", "empty")
tfiles("statement-fails-alone", "bad_transform", "broken", "result-model")

# Status records
TARGET_DOC = envelope(GOOD_T, secret)
TARGET = base64.b64decode(TARGET_DOC["payload"])
RECEIPT_PAYLOAD = base64.b64decode(signed["payload"])
RESULT_ROOT = GOOD_T["predicate"]["result"]["weights_root"]


def record(action="withdraw", target=None, root=RESULT_ROOT, **over):
    r = at.status(action=action, target=target or at.statement_digest(TARGET), weights_root=root, reason="wrong base",
                  meaning="authored", time=TIME2, replacement=("sha256:" + "5" * 64) if action == "supersede" else None)
    r["predicate"]["tool"]["version"] = "0.1.0"
    r["predicate"].update(over)
    return r


def status_doc(name, rec, expect, key="ed25519", public=True):
    (FIX / name).write_text(json.dumps(keyed(rec, key)))
    case("status", name.removesuffix(".json"), {"code": expect}, file=f"fixtures/{name}",
         **({"public_hex": public_for(key)} if public and public_for(key) else {}))


status_doc("status-withdraw.json", record(), "ok")
status_doc("status-supersede-p256.json", record("supersede"), "ok", "p256")
status_doc("status-unsigned.json", record(), "ok", "none", public=False)
(FIX / "status-wrong-key.json").write_text(json.dumps(p256_envelope(record(), P256_OTHER)))
case("status", "status-wrong-key", {"code": "bad_signature"}, file="fixtures/status-wrong-key.json", public_hex=der(P256).hex())
status_doc("status-wrong-predicate-type.json", {**record(), "predicateType": at.PREDICATE_TYPE}, "bad_type")
withdraw_with_replacement = record()
withdraw_with_replacement["predicate"]["replacement"] = "sha256:" + "5" * 64
status_doc("status-withdraw-names-a-replacement.json", withdraw_with_replacement, "bad_status")
no_replacement = record("supersede")
del no_replacement["predicate"]["replacement"]
status_doc("status-supersede-without-replacement.json", no_replacement, "bad_status")
status_doc("status-supersede-itself.json", record("supersede", replacement=at.statement_digest(TARGET)), "bad_status")
status_doc("status-unknown-action.json", record(action="revoke"), "bad_status")
status_doc("status-empty-reason.json", record(reason=""), "bad_status")
status_doc("status-meaning-outside-vocabulary.json", record(meaning="blessed"), "bad_status")
status_doc("status-target-not-a-digest.json", record(target="nope"), "bad_status")
status_doc("status-unknown-member.json", record(surprise=1), "bad_status")
status_doc("status-x-member-ok.json", record(**{"x-ticket": "42"}), "ok")
status_doc("status-time-malformed.json", record(time="soon"), "bad_status")
wrong_subject = record()
wrong_subject["subject"][0]["digest"][at.WEIGHTS_KEY] = "0" * 64
status_doc("status-subject-differs.json", wrong_subject, "subject_mismatch")

(FIX / "target-receipt.json").write_text(json.dumps(signed))
(FIX / "target-transformation.json").write_text(json.dumps(TARGET_DOC))
(FIX / "status-for-receipt.json").write_text(json.dumps(envelope(
    record("supersede", target=at.statement_digest(RECEIPT_PAYLOAD), root=claim["result_root"]), secret)))
(FIX / "status-for-another-statement.json").write_text(json.dumps(envelope(record(target="sha256:" + "6" * 64), secret)))
(FIX / "status-for-another-model.json").write_text(json.dumps(envelope(record(root="sha256:" + "6" * 64), secret)))


def sbind(id, expect, status="status-withdraw", target="target-transformation"):
    case("statusbinding", id, {"code": expect}, status=f"fixtures/{status}.json", target=f"fixtures/{target}.json")


sbind("withdraws-a-transformation", "ok")
sbind("supersedes-a-receipt", "ok", "status-for-receipt", "target-receipt")
sbind("names-another-statement", "binding_mismatch", "status-for-another-statement")
sbind("names-another-model", "binding_mismatch", "status-for-another-model")
sbind("a-receipt-is-not-the-transformation", "binding_mismatch", "status-withdraw", "target-receipt")
sbind("record-fails-alone", "bad_status", "status-empty-reason")
sbind("target-fails-alone", "bad_type", "status-withdraw", "receipt-bad-type")
sbind("target-is-a-bad-transformation", "bad_transform", "status-withdraw", "bind-broken")

for n in (0, 1, 2, 3, 4, 5, 7, 8, 9):
    case("rowtree", f"leaves-{n}", leaves_hex=[hashlib.sha256(bytes([i])).hexdigest() for i in range(n)])

shapes = model("rows-shapes", {
    "grid": rng.integers(0, 65535, (70, 33), dtype=np.uint16), "tall": rng.integers(0, 65535, (3000, 2), dtype=np.uint16),
    "vec": rng.integers(0, 65535, (5000,), dtype=np.uint16), "wide": rng.integers(0, 65535, (3, 2100), dtype=np.uint16),
    "one": np.array(7, dtype=np.uint16), "void": np.zeros((3, 0), dtype=np.uint16), "bytes": c,
})
case("rowroot", "one-shard", model="fixtures/base")
case("rowroot", "two-shards-same-root", model="fixtures/base-resharded")
case("rowroot", "different", model="fixtures/other")
case("rowroot", "awkward-shapes", model="fixtures/rows-shapes")
case("rowroot", "same-tensor-in-two-files", {"code": "bad_safetensors"}, model="fixtures/duplicate-name")

ROWS = FIX / "rows"
ROWS.mkdir()
shape_ck = Checkpoint.open(str(shapes))
shape_index = build_index(shape_ck)


def proof_case(id, ranges, name, change=None, data=None, expect="ok"):
    proof = prove(shape_index, name, ranges)
    body = fetch(shape_ck.tensors[name], proof)
    if change:
        change(proof)
    body = body if data is None else data(body)
    (ROWS / f"{id}.json").write_text(json.dumps(proof))
    (ROWS / f"{id}.data").write_bytes(body)
    case("rowproof", id, {"code": expect}, proof=f"fixtures/rows/{id}.json", data=f"fixtures/rows/{id}.data")


def flip(body):
    return bytes([body[0] ^ 1]) + body[1:]


proof_case("two-leaves", [(60, 66)], "grid")
proof_case("partial-last-leaf", [(69, 70)], "grid")
proof_case("three-ranges", [(5, 6), (2000, 2001), (2999, 3000)], "tall")
proof_case("vector-tail", [(4999, 5000)], "vec")
proof_case("one-row-per-leaf", [(2, 3)], "wide")
proof_case("scalar", [(0, 1)], "one")
proof_case("bytes", [(1, 4)], "bytes")
proof_case("flipped-bit", [(60, 66)], "grid", data=flip, expect="proof_mismatch")
proof_case("short-data", [(60, 66)], "grid", data=lambda b: b[:-1], expect="bad_proof")
proof_case("other-root", [(60, 66)], "grid", lambda p: p.update(rows_root="sha256:" + "1" * 64), expect="proof_mismatch")
proof_case("sibling-changed", [(1000, 1030)], "tall", lambda p: p["tree"].__setitem__(0, "sha256:" + "2" * 64), expect="proof_mismatch")
proof_case("path-changed", [(60, 66)], "grid", lambda p: p["path"].__setitem__(0, "sha256:" + "3" * 64), expect="proof_mismatch")
proof_case("path-short", [(60, 66)], "grid", lambda p: p["path"].pop(), expect="bad_proof")
proof_case("tree-long", [(1000, 1030)], "tall", lambda p: p["tree"].append(p["tree"][0]), expect="bad_proof")
proof_case("other-name", [(60, 66)], "grid", lambda p: p["tensor"].update(name="tall"), expect="proof_mismatch")
proof_case("other-dtype", [(60, 66)], "grid", lambda p: p["tensor"].update(dtype="F16"), expect="proof_mismatch")
proof_case("other-position", [(60, 66)], "grid", lambda p: p.update(index=p["index"] ^ 1), expect="proof_mismatch")
proof_case("range-past-end", [(60, 66)], "grid", lambda p: p.update(ranges=[[60, 71]]), expect="bad_proof")
proof_case("ranges-unsorted", [(5, 6), (2000, 2001)], "tall", lambda p: p["ranges"].reverse(), expect="bad_proof")
proof_case("unknown-member", [(60, 66)], "grid", lambda p: p.update(extra=1), expect="bad_proof")
(ROWS / "not-json.json").write_text("{")
(ROWS / "not-json.data").write_bytes(b"")
case("rowproof", "not-json", {"code": "bad_proof"}, proof="fixtures/rows/not-json.json", data="fixtures/rows/not-json.data")


# Cases found by fuzz/differential.py, where the Python and Go verifiers first disagreed. The spec text is the
# authority for each expected result.
def raw_file(path, header, data=b""):
    """A safetensors file from header text, so that malformed headers can be written on purpose."""
    text = header if isinstance(header, bytes) else json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(text)) + text + data)


def entry_of(dtype, shape, start, stop):
    return {"dtype": dtype, "shape": shape, "data_offsets": [start, stop]}


def run_root(model_dir):
    return _run(ROOT, {"kind": "root", "model": model_dir})["root"]


def run_rowroot(model_dir):
    return _run(ROOT, {"kind": "rowroot", "model": model_dir})["root"]


base_root_value, base_rows_value = run_root("fixtures/base"), run_rowroot("fixtures/base")
hidden = model("hidden-file", {"w": a, "é.bias": b, "z": c})
raw_file(hidden / ".ghost.safetensors", {"ghost": entry_of("U8", [1], 0, 1)}, b"x")
nested = model("dir-named-safetensors", {"w": a, "é.bias": b, "z": c})
(nested / "folder.safetensors").mkdir()
(nested / "folder.safetensors" / ".keep").write_text("")
for kind, expected in (("root", base_root_value), ("rowroot", base_rows_value)):
    case(kind, "dot-file-is-not-a-weight-file", {"root": expected}, model="fixtures/hidden-file")
    case(kind, "directory-named-like-a-shard-is-skipped", {"root": expected}, model="fixtures/dir-named-safetensors")

split = model("consolidated-name", {"w": a, "é.bias": b}, {"z": c})
(FIX / "consolidated-name" / "model-1.safetensors").rename(FIX / "consolidated-name" / "consolidated.safetensors")
case("root", "consolidated-is-an-ordinary-shard", {"root": base_root_value}, model="fixtures/consolidated-name")
case("rowroot", "consolidated-is-an-ordinary-shard", {"root": base_rows_value}, model="fixtures/consolidated-name")

dup_top = model("header-repeated-key", {"w": a})
good_header = (dup_top / "model-0.safetensors").read_bytes()
hlen = struct.unpack("<Q", good_header[:8])[0]
header_text = good_header[8:8 + hlen].decode().rstrip()
payload_bytes = good_header[8 + hlen:]
raw_file(dup_top / "model-0.safetensors", (header_text[:-1] + "," + header_text[1:-1] + "}").encode(), payload_bytes)
case("root", "header-repeats-a-tensor-name", {"code": "bad_safetensors"}, model="fixtures/header-repeated-key")
case("rowroot", "header-repeats-a-tensor-name", {"code": "bad_safetensors"}, model="fixtures/header-repeated-key")
dup_field = model("header-repeated-field", {"w": a})
raw_file(dup_field / "model-0.safetensors", header_text.replace('"dtype":"BF16"', '"dtype":"BF16","dtype":"BF16"').encode(), payload_bytes)
case("root", "header-repeats-a-field", {"code": "bad_safetensors"}, model="fixtures/header-repeated-field")

good_blob = (FIX / "good.neyra").read_bytes()
good_len = struct.unpack("<Q", good_blob[:8])[0]
good_header_doc = json.loads(good_blob[8:8 + good_len])
good_payload = good_blob[8 + good_len:]
for name, change in (("metadata-is-an-empty-list", lambda h: h.update({"__metadata__": []})),
                     ("metadata-is-null", lambda h: h.update({"__metadata__": None})),
                     ("metadata-is-zero", lambda h: h.update({"__metadata__": 0})),
                     ("metadata-is-a-string", lambda h: h.update({"__metadata__": ""}))):
    doc = json.loads(json.dumps(good_header_doc))
    change(doc)
    raw_file(FIX / f"{name}.neyra", doc, good_payload)
    case("patch", name, {"code": "bad_safetensors"}, file=f"fixtures/{name}.neyra")
dup_patch = good_blob[8:8 + good_len].decode().rstrip()
raw_file(FIX / "patch-header-repeats-payload.neyra", (dup_patch[:-1] + ',"0.idx":' + json.dumps(good_header_doc["0.idx"]) + "}").encode(), good_payload)
case("patch", "header-repeats-a-payload-name", {"code": "bad_safetensors"}, file="fixtures/patch-header-repeats-payload.neyra")

doc = json.loads(json.dumps(good_header_doc))
doc["__metadata__"]["neyra.meta"] = "this is a note, not JSON"
raw_file(FIX / "free-text-meta.neyra", doc, good_payload)
case("patch", "meta-is-free-text", {"code": "ok"}, file="fixtures/free-text-meta.neyra")
case("patch", "meta-is-free-text-with-base", {"code": "ok"}, file="fixtures/free-text-meta.neyra", base="fixtures/base")


def manifest_text(name, text, expect):
    doc = json.loads(json.dumps(good_header_doc))
    doc["__metadata__"]["neyra.manifest"] = text
    raw_file(FIX / f"{name}.neyra", doc, good_payload)
    case("patch", name, {"code": expect}, file=f"fixtures/{name}.neyra")


manifest_text("manifest-is-an-array", "[]", "bad_type")
manifest_text("manifest-is-a-string", '"neyra.patch/1"', "bad_type")
manifest_text("manifest-is-a-spaced-array", "[ 1 ]", "not_canonical")
manifest_text("manifest-is-a-float", "1.5", "bad_json")
tampered("shape-has-a-list.neyra", lambda d: d["tensors"][0].update({"shape": [[1]]}), expect="bad_manifest")
tampered("shape-has-a-boolean.neyra", lambda d: d["tensors"][0].update({"shape": [True]}), expect="bad_manifest")
tampered("shape-is-negative.neyra", lambda d: d["tensors"][0].update({"shape": [-1]}), expect="bad_manifest")
tampered("name-is-a-number.neyra", lambda d: d["tensors"][0].update({"name": 7}), expect="bad_manifest")
tampered("dtype-is-a-list.neyra", lambda d: d["tensors"][0].update({"dtype": ["BF16"]}), expect="bad_manifest")

reseal_head = GENESIS


def sealed(rows):
    """Entries chained from the genesis, with each prev recomputed after the edit."""
    out, previous = [], GENESIS
    for row in rows:
        row = dict(row, prev=previous)
        out.append(row)
        previous = entry_digest(row)
    return out


def second(**change):
    return sealed([entries[0], {**entries[1], **change}, entries[2]])


ledger("ledger-kind-is-a-number.jsonl", [jcs(x) for x in second(kind=7)], expect={"code": "bad_chain"})
ledger("ledger-body-is-a-list.jsonl", [jcs(x) for x in second(body=[])], expect={"code": "bad_chain"})
ledger("ledger-body-is-null.jsonl", [jcs(x) for x in second(body=None)], expect={"code": "bad_chain"})
ledger("ledger-seq-is-true.jsonl", [jcs(x) for x in second(seq=True)], expect={"code": "bad_chain"})
ledger("ledger-seq-is-a-string.jsonl", [jcs(x) for x in second(seq="1")], expect={"code": "bad_chain"})
ledger("ledger-time-in-other-digits.jsonl", [jcs(x) for x in second(time="\u0662\u0660\u0662\u0666-11-04T12:01:00Z")],
       expect={"code": "bad_chain"})
ledger("ledger-time-fullwidth-digits.jsonl", [jcs(x) for x in second(time="\uff12\uff10\uff12\uff16-11-04T12:01:00Z")],
       expect={"code": "bad_chain"})
ledger("ledger-time-newline.jsonl", [jcs(x) for x in second(time="2026-11-04T12:01:00Z\n")], expect={"code": "bad_chain"})

payload_raw = jcs(statement)
sig_entry = signed["signatures"][0]
sig_raw = base64.b64decode(sig_entry["sig"])
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
sig_text = sig_entry["sig"]
sig_unused_bits = sig_text[:-3] + alphabet[alphabet.index(sig_text[-3]) ^ 1] + sig_text[-2:]
assert base64.b64decode(sig_unused_bits) == sig_raw
receipt("receipt-signature-with-stray-bits.json", {**signed, "signatures": [{**sig_entry, "sig": sig_unused_bits}]}, "bad_signature", pub)
padded = dict(claim, note="x")
while len(jcs(statement_for(padded, good.id, head, 3))) % 3 != 1:
    padded["note"] += "x"
padded_statement = statement_for(padded, good.id, head, 3)
padded_envelope = envelope(padded_statement, secret)
payload_text = padded_envelope["payload"]
assert payload_text.endswith("==")
payload_unused_bits = payload_text[:-3] + alphabet[alphabet.index(payload_text[-3]) ^ 1] + payload_text[-2:]
receipt("receipt-padded-payload-ok.json", padded_envelope, "ok", pub)
receipt("receipt-payload-with-stray-bits.json", {**padded_envelope, "payload": payload_unused_bits}, "bad_json", pub)
bad_sig = {"keyid": sig_entry["keyid"], "sig": base64.b64encode(bytes(64)).decode()}
receipt("receipt-bad-signature-then-good.json", {**signed, "signatures": [bad_sig, sig_entry]}, "ok", pub)
receipt("receipt-good-signature-then-bad.json", {**signed, "signatures": [sig_entry, bad_sig]}, "ok", pub)
receipt("receipt-only-a-bad-signature.json", {**signed, "signatures": [bad_sig]}, "bad_signature", pub)
receipt("receipt-signatures-missing.json", {k: v for k, v in signed.items() if k != "signatures"}, "bad_json")
receipt("receipt-signatures-not-a-list.json", {**signed, "signatures": sig_entry}, "bad_json")
receipt("receipt-signature-entry-not-an-object.json", {**signed, "signatures": [1, sig_entry]}, "bad_json")
receipt("receipt-signature-without-sig.json", {**signed, "signatures": [{"keyid": sig_entry["keyid"]}]}, "bad_json")
receipt("receipt-signature-sig-not-a-string.json", {**signed, "signatures": [{**sig_entry, "sig": 5}]}, "bad_json")
receipt("receipt-payload-not-a-string.json", {**signed, "payload": 5}, "bad_json")
receipt("receipt-payload-type-wrong-and-signatures-missing.json",
        {"payloadType": "text/plain", "payload": signed["payload"]}, "bad_json")
receipt("receipt-is-an-array.json", [signed], "bad_json")
(FIX / "receipt-invalid-utf8.json").write_bytes(b'{"payloadType":"\xff"}')
case("receipt", "receipt-invalid-utf8", {"code": "bad_json"}, file="fixtures/receipt-invalid-utf8.json")
numeric = json.loads(jcs(statement))
numeric["predicate"]["claim"]["patch_id"] = 7
numeric["subject"][1]["digest"]["sha256"] = "7"
receipt("receipt-claim-digest-is-a-number.json", envelope(numeric, secret), "subject_mismatch", pub)
bare = json.loads(jcs(statement))
bare["predicate"]["claim"]["result_root"] = good.result_root.split(":", 1)[1]
receipt("receipt-claim-digest-without-prefix.json", envelope(bare, secret), "subject_mismatch", pub)

one_entry = [jcs(entries[0])]
ledger("ledger-one-entry.jsonl", one_entry)
first_head = entry_digest(entries[0])
for id, length in (("ledger-len-true", True), ("ledger-len-one-ok", 1), ("ledger-len-string", "1")):
    receipt(f"receipt-{id}.json", envelope(statement_for(claim, good.id, first_head, length), secret), "ok", pub)
binding("ledger-len-true-for-one-entry", "binding_mismatch", "fixtures/receipt-ledger-len-true.json", "fixtures/ledger-one-entry.jsonl")
binding("ledger-len-one-for-one-entry", "ok", "fixtures/receipt-ledger-len-one-ok.json", "fixtures/ledger-one-entry.jsonl")
binding("ledger-len-string", "binding_mismatch", "fixtures/receipt-ledger-len-string.json", "fixtures/ledger-one-entry.jsonl")

for c_ in cases:
    if c_["expect"] is None or (c_["kind"] == "ledger" and c_["expect"] is None) or c_["kind"] == "ledger" and c_["expect"] == {"code": "ok"}:
        c_["expect"] = _run(ROOT, c_)

(ROOT / "cases.json").write_text(json.dumps(cases, indent=1, ensure_ascii=True) + "\n")
print(len(cases), "cases")
