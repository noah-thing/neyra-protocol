"""delta.submission.v1: a directory of manifest.json and weights.safetensors, as Delta League reads it.

Arithmetic follows delta_cli.artifacts.reconstruct: every sum is float32 and is rounded once to the
parent's dtype. The parent content hash needs an identity string only the server holds, so it is
checked when the caller passes it and recorded as unchecked otherwise.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import Path

import numpy as np

from ..numerics import from_float, is_float, to_float
from ..patch import Patch, write_safetensors
from ..store import Checkpoint, LocalBlob, parse_header
from . import Unsupported, build

SCHEMA = "delta.submission.v1"
RECONSTRUCTION = "delta-reconstruct-v1"
KINDS = {"sparse-diff", "dense-diff", "lora", "full"}
WEIGHTS = "weights.safetensors"
TORCH_NAME = {"BF16": "torch.bfloat16", "F16": "torch.float16", "F32": "torch.float32", "F64": "torch.float64",
              "I64": "torch.int64", "I32": "torch.int32", "I16": "torch.int16", "I8": "torch.int8",
              "U8": "torch.uint8", "BOOL": "torch.bool"}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(ck: Checkpoint, identity: str, overrides: dict[str, np.ndarray] | None = None) -> str:
    """Delta League's canonical hash of a model: identity, then each tensor's name, shape, dtype and raw bytes."""
    h = hashlib.sha256(canonical({"schema": "delta-content-v1", "identity": identity}))
    for name in sorted(ck.tensors):
        ref = ck.tensors[name]
        data = ref.read() if overrides is None or name not in overrides else overrides[name]
        header = canonical([name, list(ref.shape), TORCH_NAME[ref.dtype]])
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        h.update(np.ascontiguousarray(data).tobytes())
    return h.hexdigest()


def check_manifest(m: dict) -> None:
    if not isinstance(m, dict) or m.get("schema_version") != SCHEMA:
        raise Unsupported("not a delta.submission.v1 manifest")
    if m.get("format") not in KINDS:
        raise Unsupported(f"format {m.get('format')!r} is not one of {sorted(KINDS)}")
    if m.get("reconstruction_version") != RECONSTRUCTION:
        raise Unsupported(f"reconstruction {m.get('reconstruction_version')!r} is not {RECONSTRUCTION}")
    try:
        uuid.UUID(str(m["parent_version_id"]))
    except (KeyError, ValueError) as e:
        raise Unsupported("parent_version_id is not a UUID") from e
    if not re.fullmatch(r"[0-9a-f]{64}", str(m.get("parent_content_hash", ""))):
        raise Unsupported("parent_content_hash is not 64 lowercase hex characters")
    if m.get("operation", "replace") not in {"replace", "add"}:
        raise Unsupported(f"operation {m.get('operation')!r}")
    files = m.get("artifacts")
    if not isinstance(files, list) or len(files) != 1 or files[0].get("name") != WEIGHTS:
        raise Unsupported("expected exactly one artifact, weights.safetensors")


def f32(bits: np.ndarray, dtype: str) -> np.ndarray:
    if dtype in ("I64", "I32", "I16", "I8", "U8", "BOOL"):
        return bits.astype(np.float32)
    return to_float(bits, dtype).astype(np.float32)


def cast(values: np.ndarray, dtype: str) -> np.ndarray:
    if not is_float(dtype):
        raise Unsupported(f"result dtype {dtype} is not a float")
    return from_float(values.astype(np.float32), dtype)


def read_artifact(path: Path) -> dict:
    refs, _ = parse_header(LocalBlob(path))
    return refs


def load(path: str | Path, base: Checkpoint, identity: str | None = None) -> Patch:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    check_manifest(manifest)
    art = manifest["artifacts"][0]
    weights = root / WEIGHTS
    if weights.stat().st_size != art["size_bytes"] or file_sha256(weights) != art["sha256"]:
        raise Unsupported("weights.safetensors does not match the manifest's size and sha256")
    parent = "unchecked, the identity string is not public"
    if identity is not None:
        if content_hash(base, identity) != manifest["parent_content_hash"]:
            raise Unsupported("the base is not the parent this submission names")
        parent = "verified"

    refs = read_artifact(weights)
    kind, add = manifest["format"], manifest.get("operation", "replace") == "add"
    mapping = manifest["tensors"]
    if kind == "full":
        mapping = {n: {"values": n} for n in refs}
        if set(refs) != set(base.tensors):
            raise Unsupported("a full submission must hold exactly the parent's tensors")
    used, results = set(), {}

    def values(key: str) -> tuple[np.ndarray, str]:
        used.add(key)
        r = refs[key]
        arr = r.read()
        if arr.dtype.kind == "u" and r.dtype in ("BF16", "F16", "F32", "F64"):
            arr = f32(arr, r.dtype)
        elif r.dtype == "I64":
            arr = arr.view(np.int64)
        return arr, r.dtype

    for name, spec in mapping.items():
        ref = base.tensors.get(name)
        if ref is None:
            raise Unsupported(f"{name}: not a tensor of the parent")
        old = ref.read()
        if kind == "lora":
            a, _ = values(spec["a"])
            b, _ = values(spec["b"])
            alpha = float(spec["alpha"])
            if a.ndim != 2 or b.ndim != 2 or old.ndim != 2 or a.shape[0] != b.shape[1] or not 1 <= a.shape[0] <= 256:
                raise Unsupported(f"{name}: bad LoRA rank")
            if not math.isfinite(alpha) or (b.shape[0], a.shape[1]) != old.shape:
                raise Unsupported(f"{name}: bad LoRA shape")
            scale = np.float32(alpha / a.shape[0])
            out = f32(old, ref.dtype) + (b.astype(np.float32) @ a.astype(np.float32)) * scale
        elif kind == "sparse-diff":
            idx, _ = values(spec["indices"])
            vals, _ = values(spec["values"])
            if idx.dtype != np.int64 or idx.ndim != 1 or vals.ndim != 1 or idx.size != vals.size:
                raise Unsupported(f"{name}: bad sparse shapes")
            if idx.size and (idx[0] < 0 or idx[-1] >= old.size or np.any(np.diff(idx) <= 0)):
                raise Unsupported(f"{name}: sparse indices must increase and stay inside the tensor")
            out = f32(old, ref.dtype).reshape(-1)
            out[idx] = out[idx] + vals.astype(np.float32) if add else vals.astype(np.float32)
            out = out.reshape(old.shape)
        else:
            vals, _ = values(spec["values"])
            if vals.shape != old.shape:
                raise Unsupported(f"{name}: dense values have the wrong shape")
            out = f32(old, ref.dtype) + vals.astype(np.float32) if add and kind == "dense-diff" else vals
        if not np.isfinite(out).all():
            raise Unsupported(f"{name}: the result is not finite")
        results[name] = cast(out, ref.dtype)
    if used != set(refs):
        raise Unsupported("the artifact holds tensors the manifest does not use")
    meta = {"source": SCHEMA, "format": kind, "operation": manifest.get("operation", "replace"),
            "parent_version_id": manifest["parent_version_id"], "parent_content_hash": manifest["parent_content_hash"],
            "parent_hash_check": parent, "artifact_sha256": art["sha256"]}
    return build(base, results, meta)


def dump(patch: Patch, base: Checkpoint, out: str | Path, parent_version_id: str, parent_content_hash: str,
         model_config_id: str) -> dict:
    """Write the patch as an exact sparse-diff replacement, which any server reconstructs to the same bits."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tensors, mapping, kinds = {}, {}, set()
    for n, e in enumerate(sorted(patch.edits, key=lambda e: e.name)):
        if e.dtype not in ("BF16", "F16", "F32"):
            raise Unsupported(f"{e.name}: {e.dtype} values cannot be written as floats here")
        full = e.dense(base.tensors[e.name].read())
        i, v = f"t{n}.indices", f"t{n}.values"
        tensors[i] = full.idx.astype("<i8")
        tensors[v] = full.val.view(np.float32) if e.dtype == "F32" else full.val
        mapping[e.name] = {"indices": i, "values": v}
        if e.dtype != "F32":
            kinds.add(e.dtype)
    if not tensors:
        raise Unsupported("an empty patch is not a submission")
    if len(kinds) > 1:
        raise Unsupported("BF16 and F16 tensors cannot share one weights file, since the writer labels 16 bit bits once")
    write_safetensors(out / WEIGHTS, tensors, {"format": "pt"}, kinds.pop() if kinds else None)
    path = out / WEIGHTS
    manifest = {"schema_version": SCHEMA, "format": "sparse-diff", "operation": "replace",
                "reconstruction_version": RECONSTRUCTION, "parent_version_id": parent_version_id,
                "parent_content_hash": parent_content_hash, "model_config_id": model_config_id, "tensors": mapping,
                "artifacts": [{"name": WEIGHTS, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


__all__ = ["content_hash", "dump", "load"]
