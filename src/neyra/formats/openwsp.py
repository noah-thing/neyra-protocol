"""Edits from Sparks of Weight-Space Programming: the authors' edit/ folder and OpenWSP's versioned v1 of it.

Both are a manifest.json and an edit.npz of bfloat16 bit patterns. They publish a hash of every tensor
before and after, and v1 adds a payload hash and a content address, so every claim here is checked.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from ..patch import Patch, PatchMismatch, TensorEdit
from ..protocol import bits_digest, tensor_table
from ..store import Checkpoint, open_npz
from . import Unsupported

FORMAT = "openwsp.edit"


def sha(bits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bits, dtype="<u2").tobytes()).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def payload_sha256(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        head = canonical({"name": name, "dtype": a.dtype.str, "shape": list(a.shape)})
        h.update(len(head).to_bytes(8, "little") + head + a.nbytes.to_bytes(8, "little") + a.tobytes())
    return h.hexdigest()


def edit_sha256(manifest: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != "edit_sha256"})).hexdigest()


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as z:
        if any(i.filename.count("/") for i in z.infolist()):
            raise Unsupported(f"{path}: nested names in the archive")
    with open_npz(path) as z:
        return {k: z[k] for k in z.files}


def verify_v1(manifest: dict, arrays: dict[str, np.ndarray]) -> None:
    if manifest.get("format_version") != 1:
        raise Unsupported(f"OpenWSP format_version {manifest.get('format_version')!r} is not supported")
    if edit_sha256(manifest) != manifest.get("edit_sha256"):
        raise PatchMismatch("edit_sha256 does not match the manifest", "payload_mismatch")
    if payload_sha256(arrays) != manifest.get("payload_sha256"):
        raise PatchMismatch("payload_sha256 does not match edit.npz", "payload_mismatch")


def load(path: str | Path, base: Checkpoint) -> Patch:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    arrays = read_arrays(root / "edit.npz")
    versioned = manifest.get("format") == FORMAT
    if versioned:
        verify_v1(manifest, arrays)
    edits = []
    for i, t in enumerate(manifest["tensors"]):
        if t["form"] not in ("sparse", "lowrank"):
            raise Unsupported(f"{t['name']}: form {t['form']!r}")
        low_rank = t["form"] == "lowrank"
        ref = base.tensors.get(t["name"])
        if ref is None or ref.dtype != "BF16" or list(ref.shape) != list(t["shape"]):
            raise PatchMismatch(f"{t['name']}: not a bfloat16 tensor of this shape in the base")
        bits = ref.read()
        if sha(bits) != t["base_sha256"]:
            raise PatchMismatch(f"{t['name']}: not the base the published edit was made on")
        edit = TensorEdit(t["name"], "BF16", tuple(t["shape"]), bits_digest("BF16", bits), "",
                          arrays[f"t{i}_idx"].astype(np.int64), arrays[f"t{i}_val"].astype(np.uint16),
                          arrays[f"t{i}_a"].astype(np.float64) if low_rank else None,
                          arrays[f"t{i}_b"].astype(np.float64) if low_rank else None)
        order = np.argsort(edit.idx, kind="stable")
        edit.idx, edit.val = edit.idx[order], edit.val[order]
        rebuilt = edit.apply(bits, verify=False)
        if sha(rebuilt) != t["edited_sha256"]:
            raise PatchMismatch(f"{t['name']}: rebuilt tensor does not match the published hash", "result_mismatch")
        edit.result_digest = bits_digest("BF16", rebuilt)
        edits.append(edit)
    meta = {"source": str(root), "format": manifest.get("format", "wsp-legacy"),
            "digests": "published base and edited hashes of every tensor matched"}
    if versioned:
        meta["edit_sha256"] = manifest["edit_sha256"]
    return Patch(edits, meta).bind(tensor_table(base))


def dump(patch: Patch, base: Checkpoint, out: str | Path, base_model: str | None = None,
         revision: str | None = None) -> dict:
    """Write a v1 sparse edit. A low rank tensor is written as the entries it changes."""
    arrays, tensors = {}, []
    for e in sorted(patch.edits, key=lambda e: e.name):
        if e.dtype != "BF16":
            raise Unsupported(f"{e.name}: OpenWSP edits are bfloat16 only, this tensor is {e.dtype}")
        old = base.tensors[e.name].read()
        full = e.dense(old)
        if old.size > np.iinfo(np.int32).max:
            raise Unsupported(f"{e.name}: too large for the int32 indices the format uses")
        new = e.apply(old)
        i = len(tensors)
        arrays[f"t{i}_idx"] = full.idx.astype("<i4")
        arrays[f"t{i}_val"] = full.val.astype("<u2")
        tensors.append({"name": e.name, "shape": list(old.shape), "dtype": "bfloat16", "form": "sparse",
                        "changed_entries": len(full.idx), "base_sha256": sha(old), "edited_sha256": sha(new)})
    manifest = {"format": FORMAT, "format_version": 1, "tensors": tensors, "changed_tensors": len(tensors),
                "changed_entries_total": sum(t["changed_entries"] for t in tensors),
                "payload_sha256": payload_sha256(arrays)}
    if base_model is not None:
        manifest["base_model"] = base_model
    if revision is not None:
        manifest["base_revision"] = revision
    manifest["edit_sha256"] = edit_sha256(manifest)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "edit.npz", **arrays)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
