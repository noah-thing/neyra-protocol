"""Verify protocol objects in the order the spec fixes, returning normative error codes."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from . import attest
from .ledger import BrokenChain, entry_digest, read_entries
from .ledger import verify as verify_ledger
from .patch import (
    KNOWN_MEMBERS,
    KNOWN_TENSOR,
    LOW_RANK,
    MANIFEST_KEY,
    TYPE,
    Patch,
    PatchMismatch,
    payload_digest,
)
from .protocol import (
    ProtocolError,
    decode_envelope,
    jcs,
    open_envelope,
    root_of,
    strict_loads,
    tensor_table,
)
from .rows import build_index, text, tree_root, verify_proof
from .store import ITEMSIZE, Checkpoint, FormatError, LocalBlob, parse_header

TENSOR_MEMBERS = {"name", "dtype", "shape", "base_digest", "result_digest", "idx_digest", "val_digest"}
RAW = {1: "U8", 2: "U16", 4: "U32", 8: "U64"}
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def good_shape(value) -> bool:
    return isinstance(value, list) and all(type(d) is int and d >= 0 for d in value)


def good_digest(value) -> bool:
    return isinstance(value, str) and DIGEST.fullmatch(value) is not None


def low_rank_shape(t: dict, requires: list) -> bool:
    """Both factor digests or neither; only 2-D float tensors; and the extension must be named in `requires`."""
    if ("left_digest" in t) != ("right_digest" in t):
        return False
    if "left_digest" not in t:
        return True
    return (good_digest(t["left_digest"]) and good_digest(t["right_digest"]) and len(t["shape"]) == 2
            and t["dtype"] in ("BF16", "F16", "F32") and LOW_RANK in requires)


def factors_fit(t: dict, left, right) -> bool:
    rows, cols = t["shape"]
    return (left.dtype == right.dtype == "F64" and len(left.shape) == len(right.shape) == 2
            and left.shape[0] == rows and right.shape[1] == cols and left.shape[1] == right.shape[0] >= 1)


def check_patch(path: str | Path, base: str | Path | None = None) -> str:
    try:
        refs, meta = parse_header(LocalBlob(Path(path)))
    except (FormatError, OSError):
        return "bad_safetensors"
    if MANIFEST_KEY not in meta:
        return "bad_type"
    try:
        doc = strict_loads(meta[MANIFEST_KEY])
        if jcs(doc).decode() != meta[MANIFEST_KEY]:
            return "not_canonical"
    except ProtocolError as e:
        return e.code
    if not isinstance(doc, dict) or doc.get("type") != TYPE:
        return "bad_type"
    entries = doc.get("tensors") if isinstance(doc.get("tensors"), list) else []
    if any(k not in KNOWN_MEMBERS and not k.startswith("x-") for k in doc):
        return "unknown_member"
    for t in entries:
        if isinstance(t, dict) and any(k not in KNOWN_TENSOR and not k.startswith("x-") for k in t):
            return "unknown_member"
    if not KNOWN_MEMBERS <= set(doc) or not isinstance(doc["requires"], list) or not isinstance(doc["tensors"], list):
        return "bad_manifest"
    if not all(isinstance(x, str) for x in doc["requires"]) or not all(map(good_digest, (doc["base_root"], doc["result_root"]))):
        return "bad_manifest"
    for t in entries:
        if not isinstance(t, dict) or not TENSOR_MEMBERS <= set(t) or not isinstance(t["name"], str):
            return "bad_manifest"
        if not isinstance(t["dtype"], str) or t["dtype"] not in ITEMSIZE or not good_shape(t["shape"]):
            return "bad_manifest"
        if not all(good_digest(t[k]) for k in ("base_digest", "result_digest", "idx_digest", "val_digest")):
            return "bad_manifest"
    order = [t["name"].encode("utf-16-be", "surrogatepass") for t in entries]
    if order != sorted(set(order)):
        return "bad_manifest"
    if not all(low_rank_shape(t, doc["requires"]) for t in entries):
        return "bad_manifest"
    if set(doc["requires"]) - {LOW_RANK}:
        return "unknown_extension"
    expected = {f"{i}.{part}" for i, t in enumerate(entries) for part in
                ("idx", "val", *(("left", "right") if "left_digest" in t else ()))}
    if set(refs) != expected:
        return "bad_manifest"
    for i, t in enumerate(doc["tensors"]):
        idx, val = refs.get(f"{i}.idx"), refs.get(f"{i}.val")
        if idx is None or val is None or idx.dtype != "I64" or val.dtype != RAW[ITEMSIZE[t["dtype"]]]:
            return "bad_manifest"
        if len(idx.shape) != 1 or idx.shape != val.shape:
            return "bad_manifest"
        if "left_digest" in t and not factors_fit(t, refs[f"{i}.left"], refs[f"{i}.right"]):
            return "bad_manifest"
    for i, t in enumerate(doc["tensors"]):
        if payload_digest(refs[f"{i}.idx"].read()) != t["idx_digest"] or payload_digest(refs[f"{i}.val"].read()) != t["val_digest"]:
            return "payload_mismatch"
        if "left_digest" in t and (payload_digest(refs[f"{i}.left"].read()) != t["left_digest"]
                                   or payload_digest(refs[f"{i}.right"].read()) != t["right_digest"]):
            return "payload_mismatch"
    for i, t in enumerate(doc["tensors"]):
        flat = refs[f"{i}.idx"].read().view(np.int64)
        if flat.size and (flat[0] < 0 or flat[-1] >= math.prod(t["shape"]) or np.any(np.diff(flat) <= 0)):
            return "bad_index"
    if base is None:
        return "ok"
    patch = Patch.load(path)
    ck = Checkpoint.open(str(base), protocol=True)
    table = tensor_table(ck)
    if root_of(table) != patch.base_root:
        return "base_mismatch"
    for e in patch.edits:
        row = table.get(e.name)
        if row is None or row["dtype"] != e.dtype or tuple(row["shape"]) != e.shape or row["digest"] != e.base_digest:
            return "base_mismatch"
    try:
        patch.apply(ck)
    except PatchMismatch as e:
        return e.code
    changed = {**table, **{e.name: {**table[e.name], "digest": e.result_digest} for e in patch.edits}}
    return "ok" if root_of(changed) == patch.result_root else "result_mismatch"


def check_ledger(path: str | Path) -> tuple[str, str | None]:
    try:
        head, _ = verify_ledger(Path(path))
    except (BrokenChain, ProtocolError):
        return "bad_chain", None
    return "ok", head


def check_receipt(path: str | Path, public: bytes | None = None) -> str:
    try:
        doc = strict_loads(Path(path).read_bytes())
        open_envelope(doc, public)
    except ProtocolError as e:
        return e.code
    return "ok"


def check_binding(receipt: str | Path, ledger: str | Path, patch: str | Path) -> str:
    code = check_receipt(receipt)
    if code != "ok":
        return code
    code, _ = check_ledger(ledger)
    if code != "ok":
        return code
    code = check_patch(patch)
    if code != "ok":
        return code
    predicate = open_envelope(strict_loads(Path(receipt).read_text(encoding="utf-8"))).statement["predicate"]
    claim = predicate["claim"]
    p = Patch.load(patch)
    if p.id != claim["patch_id"] or p.result_root != claim["result_root"]:
        return "binding_mismatch"
    entries, _ = read_entries(Path(ledger))
    length = predicate["ledger_len"]
    for e in entries:
        if entry_digest(e) == predicate["ledger_head"]:
            # bool is an int in Python: `True == 1` must not satisfy a ledger of one entry.
            return "ok" if type(length) is int and length == e["seq"] + 1 else "binding_mismatch"
    return "binding_mismatch"


def check_transform(path: str | Path, public: bytes | None = None) -> str:
    try:
        attest.open_transformation(strict_loads(Path(path).read_text(encoding="utf-8")), public)
    except ProtocolError as e:
        return e.code
    return "ok"


def check_status(path: str | Path, public: bytes | None = None) -> str:
    try:
        attest.open_status(strict_loads(Path(path).read_text(encoding="utf-8")), public)
    except ProtocolError as e:
        return e.code
    return "ok"


def check_transform_binding(statement: str | Path, patch: str | Path) -> str:
    code = check_transform(statement)
    if code != "ok":
        return code
    code = check_patch(patch)
    if code != "ok":
        return code
    opened = attest.open_transformation(strict_loads(Path(statement).read_text(encoding="utf-8")))
    return attest.bind_patch(opened.statement, Patch.load(patch))


def check_transform_files(statement: str | Path, model: str | Path) -> str:
    code = check_transform(statement)
    if code != "ok":
        return code
    opened = attest.open_transformation(strict_loads(Path(statement).read_text(encoding="utf-8")))
    try:
        return attest.bind_model(opened.statement, model)
    except FormatError:
        return "bad_safetensors"


def open_any(path: str | Path) -> attest.Opened:
    """A statement of either predicate type, verified alone."""
    doc = strict_loads(Path(path).read_text(encoding="utf-8"))
    kind = decode_envelope(doc)[0].get("predicateType")
    if kind == attest.PREDICATE_TYPE:
        return attest.open_transformation(doc)
    opened = open_envelope(doc)
    return attest.Opened(opened.statement, opened.signed, decode_envelope(doc)[1])


def check_status_binding(status: str | Path, target: str | Path) -> str:
    code = check_status(status)
    if code != "ok":
        return code
    try:
        found = open_any(target)
    except ProtocolError as e:
        return e.code
    return attest.bind_target(attest.open_status(strict_loads(Path(status).read_text(encoding="utf-8"))), found)


def check_rowproof(proof: str | Path, data: str | Path) -> str:
    try:
        verify_proof(strict_loads(Path(proof).read_bytes()), Path(data).read_bytes())
    except ProtocolError as e:
        return "bad_proof" if e.code == "bad_json" else e.code
    return "ok"


def run_vectors(root: str | Path) -> list[tuple[str, str, str]]:
    """Every case in cases.json. Returns (id, expected, got) for failures."""
    root = Path(root)
    failures = []
    for case in json.loads((root / "cases.json").read_text(encoding="utf-8")):
        kind, want = case["kind"], case["expect"]
        try:
            got = _run(root, case)
        except Exception as e:  # noqa: BLE001 - a verifier that crashes on a vector has failed it
            got = {"crash": f"{type(e).__name__}: {e}"}
        if got != want:
            failures.append((f"{kind}:{case['id']}", str(want), str(got)))
    return failures


def _run(root: Path, case: dict):
    kind = case["kind"]
    if kind == "jcs":
        try:
            return {"code": "ok", "hex": jcs(strict_loads(bytes.fromhex(case["input_hex"]))).hex()}
        except ProtocolError as e:
            return {"code": e.code}
    if kind == "tensor":
        from .protocol import tensor_digest

        return {"digest": tensor_digest(case["dtype"], tuple(case["shape"]), bytes.fromhex(case["raw_hex"]))}
    if kind == "digest":
        from .protocol import digest

        return {"digest": digest(case["domain"], bytes.fromhex(case["data_hex"]))}
    if kind == "root":
        try:
            return {"root": root_of(tensor_table(Checkpoint.open(str(root / case["model"]), protocol=True)))}
        except FormatError:
            return {"code": "bad_safetensors"}
    if kind == "patch":
        base = root / case["base"] if case.get("base") else None
        return {"code": check_patch(root / case["file"], base)}
    if kind == "ledger":
        code, head = check_ledger(root / case["file"])
        return {"code": code, "head": head} if code == "ok" else {"code": code}
    if kind == "receipt":
        public = bytes.fromhex(case["public_hex"]) if case.get("public_hex") else None
        return {"code": check_receipt(root / case["file"], public)}
    if kind == "rowtree":
        leaves = b"".join(bytes.fromhex(h) for h in case["leaves_hex"])
        return {"root": text(tree_root(leaves))}
    if kind == "rowroot":
        try:
            return {"root": build_index(Checkpoint.open(str(root / case["model"]), protocol=True)).rows_root}
        except FormatError:
            return {"code": "bad_safetensors"}
    if kind == "rowproof":
        return {"code": check_rowproof(root / case["proof"], root / case["data"])}
    if kind in ("transform", "status"):
        public = bytes.fromhex(case["public_hex"]) if case.get("public_hex") else None
        return {"code": (check_transform if kind == "transform" else check_status)(root / case["file"], public)}
    if kind == "transformbinding":
        return {"code": check_transform_binding(root / case["statement"], root / case["patch"])}
    if kind == "transformfiles":
        return {"code": check_transform_files(root / case["statement"], root / case["model"])}
    if kind == "statusbinding":
        return {"code": check_status_binding(root / case["status"], root / case["target"])}
    if kind == "binding":
        return {"code": check_binding(root / case["receipt"], root / case["ledger"], root / case["patch"])}
    raise ValueError(f"unknown case kind {kind}")
