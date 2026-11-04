"""Verify protocol objects in the order the spec fixes, returning normative error codes."""

from __future__ import annotations

import json
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
from .store import Checkpoint, FormatError, LocalBlob, parse_header


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
    if any(k not in KNOWN_MEMBERS and not k.startswith("x-") for k in doc):
        return "unknown_member"
    if not KNOWN_MEMBERS <= set(doc):
        return "bad_type"
    if set(doc["requires"]) - {LOW_RANK}:
        return "unknown_extension"
    for i, t in enumerate(doc["tensors"]):
        if any(k not in KNOWN_TENSOR and not k.startswith("x-") for k in t):
            return "unknown_member"
        idx, val = refs.get(f"{i}.idx"), refs.get(f"{i}.val")
        if idx is None or val is None or idx.dtype != "I64":
            return "payload_mismatch"
        if payload_digest(idx.read()) != t["idx_digest"] or payload_digest(val.read()) != t["val_digest"]:
            return "payload_mismatch"
        flat = idx.read().view(np.int64)
        size = int(np.prod(t["shape"], dtype=np.int64))
        if flat.size and (flat[0] < 0 or flat[-1] >= size or np.any(np.diff(flat) <= 0)):
            return "bad_index"
    if base is None:
        return "ok"
    patch = Patch.load(path)
    ck = Checkpoint.open(str(base))
    table = tensor_table(ck)
    if root_of(table) != patch.base_root:
        return "base_mismatch"
    try:
        patch.apply(ck)
        changed = {**table, **{e.name: {**table[e.name], "digest": e.result_digest} for e in patch.edits}}
    except PatchMismatch as e:
        return e.code
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
        return {"root": root_of(tensor_table(Checkpoint.open(str(root / case["model"]))))}
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
