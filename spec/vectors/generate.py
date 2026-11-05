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
from neyra.protocol import digest, envelope, jcs, statement_for, tensor_table
from neyra.rows import build_index, fetch, prove
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
