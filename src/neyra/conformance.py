"""Verify protocol objects in the order the spec fixes, returning normative error codes."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .ledger import BrokenChain
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
from .protocol import ProtocolError, jcs, open_envelope, root_of, strict_loads, tensor_table
from .store import ITEMSIZE, Checkpoint, FormatError, LocalBlob, parse_header

TENSOR_MEMBERS = {"name", "dtype", "shape", "base_digest", "result_digest", "idx_digest", "val_digest"}
RAW = {1: "U8", 2: "U16", 4: "U32", 8: "U64"}
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def good_digest(value) -> bool:
    return isinstance(value, str) and DIGEST.fullmatch(value) is not None


def check_patch(path: str | Path, base: str | Path | None = None) -> str:
    try:
        refs, meta = parse_header(LocalBlob(Path(path)))
    except (FormatError, OSError):
        return "bad_safetensors"
    if MANIFEST_KEY not in meta:
        return "bad_type"
    try:
        doc = strict_loads(meta[MANIFEST_KEY])
        if not isinstance(doc, dict):
            return "bad_json"
        if jcs(doc).decode() != meta[MANIFEST_KEY]:
            return "not_canonical"
    except ProtocolError as e:
        return e.code
    if doc.get("type") != TYPE:
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
        if not isinstance(t, dict) or not TENSOR_MEMBERS <= set(t) or t["dtype"] not in ITEMSIZE or not isinstance(t["shape"], list):
            return "bad_manifest"
        if not all(good_digest(t[k]) for k in ("base_digest", "result_digest", "idx_digest", "val_digest")):
            return "bad_manifest"
    order = [t["name"].encode("utf-16-be", "surrogatepass") for t in entries]
    if order != sorted(set(order)):
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
    for i, t in enumerate(doc["tensors"]):
        if payload_digest(refs[f"{i}.idx"].read()) != t["idx_digest"] or payload_digest(refs[f"{i}.val"].read()) != t["val_digest"]:
            return "payload_mismatch"
    for i, t in enumerate(doc["tensors"]):
        flat = refs[f"{i}.idx"].read().view(np.int64)
        if flat.size and (flat[0] < 0 or flat[-1] >= math.prod(t["shape"]) or np.any(np.diff(flat) <= 0)):
            return "bad_index"
    if base is None:
        return "ok"
    patch = Patch.load(path)
    ck = Checkpoint.open(str(base))
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
        doc = strict_loads(Path(path).read_text())
        open_envelope(doc, public)
    except ProtocolError as e:
        return e.code
    return "ok"


def run_vectors(root: str | Path) -> list[tuple[str, str, str]]:
    """Every case in cases.json. Returns (id, expected, got) for failures."""
    root = Path(root)
    failures = []
    for case in json.loads((root / "cases.json").read_text()):
        kind, want = case["kind"], case["expect"]
        got = _run(root, case)
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
            return {"root": root_of(tensor_table(Checkpoint.open(str(root / case["model"]))))}
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
    raise ValueError(f"unknown case kind {kind}")
