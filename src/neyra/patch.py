"""A weight edit as a small, exact, replayable file.

A patch is itself a safetensors file, so reading one cannot execute code. Its identity
is the digest of a canonical manifest, which binds the whole base model, the whole
result, and every byte of payload.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .numerics import from_float, is_float, to_float
from .protocol import Table, bits_digest, digest, jcs, root_of, strict_loads, tensor_table
from .store import ITEMSIZE, Checkpoint, LocalBlob, parse_header

TYPE = "neyra.patch/1"
LOW_RANK = "low-rank/1"
MANIFEST_KEY = "neyra.manifest"
META_KEY = "neyra.meta"
KNOWN_MEMBERS = {"type", "base_root", "result_root", "requires", "tensors"}
KNOWN_TENSOR = {"name", "dtype", "shape", "base_digest", "result_digest", "idx_digest", "val_digest",
                "left_digest", "right_digest"}
RAW_NAMES = {np.dtype(np.uint8): "U8", np.dtype(np.uint16): "U16", np.dtype(np.uint32): "U32",
             np.dtype(np.uint64): "U64", np.dtype(np.int32): "I32", np.dtype(np.int64): "I64",
             np.dtype(np.float32): "F32", np.dtype(np.float64): "F64"}


class PatchMismatch(ValueError):
    def __init__(self, message: str, code: str = "base_mismatch"):
        super().__init__(message)
        self.code = code


def payload_digest(arr: np.ndarray) -> str:
    return digest("payload", np.ascontiguousarray(arr).tobytes())


@dataclass
class TensorEdit:
    name: str
    dtype: str
    shape: tuple[int, ...]
    base_digest: str
    result_digest: str
    idx: np.ndarray
    val: np.ndarray
    left: np.ndarray | None = None
    right: np.ndarray | None = None

    @property
    def low_rank(self) -> bool:
        return self.left is not None

    def apply(self, base_bits: np.ndarray, verify: bool = True) -> np.ndarray:
        if verify and bits_digest(self.dtype, base_bits) != self.base_digest:
            raise PatchMismatch(f"{self.name}: this is not the tensor the edit was made on")
        if self.low_rank:
            w = to_float(base_bits, self.dtype).astype(np.float64)
            for k in range(self.left.shape[1]):
                w = w + np.outer(self.left[:, k], self.right[k])
            out = from_float(w.astype(np.float32), self.dtype).reshape(-1)
        else:
            out = np.array(base_bits, copy=True).reshape(-1)
        if self.idx.size and (self.idx[0] < 0 or self.idx[-1] >= out.size or np.any(np.diff(self.idx) <= 0)):
            raise PatchMismatch(f"{self.name}: indices must be increasing and inside the tensor", "bad_index")
        out[self.idx] = self.val
        out = out.reshape(self.shape)
        if verify and bits_digest(self.dtype, out) != self.result_digest:
            raise PatchMismatch(f"{self.name}: rebuilt tensor does not match the recorded digest", "result_mismatch")
        return out

    def dense(self, base_bits: np.ndarray) -> TensorEdit:
        """The same edit with every changed entry listed, so it can be split."""
        if not self.low_rank:
            return self
        edited = self.apply(base_bits, verify=False)
        idx = np.flatnonzero(base_bits.reshape(-1) != edited.reshape(-1))
        return TensorEdit(self.name, self.dtype, self.shape, self.base_digest, self.result_digest,
                          idx, edited.reshape(-1)[idx])

    def entry(self) -> dict:
        out = {"name": self.name, "dtype": self.dtype, "shape": list(self.shape), "base_digest": self.base_digest,
               "result_digest": self.result_digest, "idx_digest": payload_digest(self.idx.astype("<i8")),
               "val_digest": payload_digest(self.val)}
        if self.low_rank:
            out["left_digest"] = payload_digest(self.left)
            out["right_digest"] = payload_digest(self.right)
        return out


@dataclass
class Patch:
    edits: list[TensorEdit]
    meta: dict = field(default_factory=dict)
    base_root: str | None = None
    result_root: str | None = None

    @property
    def entries(self) -> int:
        return sum(len(e.idx) for e in self.edits)

    def bind(self, table: Table) -> Patch:
        """Tie this patch to a whole base model, not only the tensors it touches."""
        self.base_root = root_of(table)
        changed = dict(table)
        for e in self.edits:
            if table[e.name]["digest"] != e.base_digest:
                raise PatchMismatch(f"{e.name}: this is not the tensor the edit was made on")
            changed[e.name] = {**table[e.name], "digest": e.result_digest}
        self.result_root = root_of(changed)
        return self

    def manifest(self) -> dict:
        if self.base_root is None:
            raise PatchMismatch("this patch is not bound to a base model yet")
        requires = [LOW_RANK] if any(e.low_rank for e in self.edits) else []
        return {"type": TYPE, "base_root": self.base_root, "result_root": self.result_root, "requires": requires,
                "tensors": [e.entry() for e in sorted(self.edits, key=lambda e: e.name.encode("utf-16-be"))]}

    @property
    def id(self) -> str:
        return digest("patch", jcs(self.manifest()))

    def apply(self, ckpt: Checkpoint, verify: bool = True) -> dict[str, np.ndarray]:
        out = {}
        for e in self.edits:
            ref = ckpt.tensors.get(e.name)
            if ref is None:
                raise PatchMismatch(f"{e.name}: not in {ckpt.spec}")
            out[e.name] = e.apply(ref.read(), verify)
        return out

    def check_base(self, ckpt: Checkpoint) -> Table:
        """Hard failure unless every tensor of the model matches the patch's base."""
        table = tensor_table(ckpt)
        if root_of(table) != self.base_root:
            raise PatchMismatch(f"{ckpt.spec} is not the base model this patch was made for")
        return table

    def invert(self, ckpt: Checkpoint) -> Patch:
        inverse = []
        for e in self.edits:
            base = ckpt.tensors[e.name].read()
            full = e.dense(base)
            inverse.append(TensorEdit(e.name, e.dtype, e.shape, e.result_digest, e.base_digest,
                                      full.idx, base.reshape(-1)[full.idx]))
        return Patch(inverse, {**self.meta, "inverse_of": self.id}, self.result_root, self.base_root)

    def save(self, path: str | Path) -> None:
        tensors: dict[str, np.ndarray] = {}
        for i, e in enumerate(sorted(self.edits, key=lambda e: e.name.encode("utf-16-be"))):
            tensors[f"{i}.idx"] = e.idx.astype("<i8")
            tensors[f"{i}.val"] = e.val
            if e.low_rank:
                tensors[f"{i}.left"] = e.left
                tensors[f"{i}.right"] = e.right
        write_safetensors(path, tensors, {MANIFEST_KEY: jcs(self.manifest()).decode(),
                                          META_KEY: json.dumps(self.meta, default=str)})

    @classmethod
    def load(cls, path: str | Path) -> Patch:
        refs, meta = parse_header(LocalBlob(Path(path)))
        if MANIFEST_KEY not in meta:
            raise PatchMismatch(f"{path}: not a Neyra patch", "bad_type")
        doc = strict_loads(meta[MANIFEST_KEY])
        if doc.get("type") != TYPE:
            raise PatchMismatch(f"{path}: {doc.get('type')} is not supported", "bad_type")
        extra = {k for k in doc if k not in KNOWN_MEMBERS and not k.startswith("x-")}
        if extra:
            raise PatchMismatch(f"{path}: unknown members {sorted(extra)}", "unknown_member")
        unknown = set(doc["requires"]) - {LOW_RANK}
        if unknown:
            raise PatchMismatch(f"{path}: needs extensions {sorted(unknown)}", "unknown_extension")

        def arr(key: str, kind=None) -> np.ndarray | None:
            ref = refs.get(key)
            if ref is None:
                return None
            return ref.read().view(kind) if kind else ref.read()

        edits = []
        for i, t in enumerate(doc["tensors"]):
            edit = TensorEdit(t["name"], t["dtype"], tuple(t["shape"]), t["base_digest"], t["result_digest"],
                              arr(f"{i}.idx", np.int64), arr(f"{i}.val"),
                              arr(f"{i}.left", np.float64), arr(f"{i}.right", np.float64))
            if payload_digest(edit.idx) != t["idx_digest"] or payload_digest(edit.val) != t["val_digest"]:
                raise PatchMismatch(f"{t['name']}: payload does not match the manifest", "payload_mismatch")
            if edit.low_rank and (payload_digest(edit.left) != t.get("left_digest")
                                  or payload_digest(edit.right) != t.get("right_digest")):
                raise PatchMismatch(f"{t['name']}: payload does not match the manifest", "payload_mismatch")
            edits.append(edit)
        extras = json.loads(meta.get(META_KEY, "{}"))
        return cls(edits, extras, doc["base_root"], doc["result_root"])


def diff_tensor(name: str, dtype: str, base: np.ndarray, edited: np.ndarray) -> TensorEdit:
    """Smallest exact encoding of edited relative to base."""
    idx = np.flatnonzero(base.reshape(-1) != edited.reshape(-1))
    a, b = bits_digest(dtype, base), bits_digest(dtype, edited.reshape(base.shape))
    sparse = TensorEdit(name, dtype, tuple(base.shape), a, b, idx, edited.reshape(-1)[idx])
    if base.ndim != 2 or not is_float(dtype) or len(idx) < 0.25 * base.size:
        return sparse

    w0 = to_float(base, dtype).astype(np.float64)
    d = to_float(edited, dtype).astype(np.float64).reshape(base.shape) - w0
    u, s, vh = np.linalg.svd(d, full_matrices=False)
    budget = len(idx) * (8 + ITEMSIZE[dtype])
    for rank in (r for r in (1, 2, 4, 8) if r <= len(s)):
        left, right = u[:, :rank] * s[:rank], vh[:rank]
        w = w0
        for k in range(rank):
            w = w + np.outer(left[:, k], right[k])
        approx = from_float(w.astype(np.float32), dtype).reshape(-1)
        wrong = np.flatnonzero(approx != edited.reshape(-1))
        size = left.nbytes + right.nbytes + len(wrong) * (8 + ITEMSIZE[dtype])
        if size < budget:
            return TensorEdit(name, dtype, tuple(base.shape), a, b, wrong, edited.reshape(-1)[wrong], left, right)
    return sparse


def make_patch(base: Checkpoint, edited: Checkpoint, meta: dict | None = None) -> Patch:
    edits = []
    for name, ref in base.tensors.items():
        other = edited.tensors.get(name)
        if other is None or other.dtype != ref.dtype or other.shape != ref.shape:
            continue
        a, b = ref.read(), other.read()
        if not np.array_equal(a, b):
            edits.append(diff_tensor(name, ref.dtype, a, b))
    return Patch(edits, meta or {}).bind(tensor_table(base))


def write_safetensors(path: str | Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None,
                      bits_as: str | None = None) -> None:
    """Write tensors. Raw float bits arrive as unsigned ints, so bits_as names their real dtype."""
    header: dict = {"__metadata__": metadata} if metadata else {}
    offset = 0
    for name, arr in tensors.items():
        kind = bits_as if bits_as and arr.dtype == np.uint16 else RAW_NAMES[arr.dtype]
        header[name] = {"dtype": kind, "shape": list(arr.shape), "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for arr in tensors.values():
            f.write(np.ascontiguousarray(arr).tobytes())


def write_checkpoint(base: Checkpoint, edited: dict[str, np.ndarray], out: str | Path) -> None:
    """Write a full model directory with edited tensors swapped in. One shard per source shard."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, list] = {}
    for ref in base.tensors.values():
        by_shard.setdefault(ref.blob.name, []).append(ref)
    for shard, refs in by_shard.items():
        refs.sort(key=lambda r: r.start)
        header: dict = {"__metadata__": {"format": "pt"}}
        offset = 0
        for r in refs:
            header[r.name] = {"dtype": r.dtype, "shape": list(r.shape), "data_offsets": [offset, offset + r.nbytes]}
            offset += r.nbytes
        blob = json.dumps(header, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)
        with open(out / shard, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for r in refs:
                bits = edited.get(r.name)
                f.write(r.read().tobytes() if bits is None else np.ascontiguousarray(bits).tobytes())
    for name in base.files:
        if not name.endswith(".safetensors") and not name.startswith(".") and "/" not in name:
            body = base.raw(name)
            if body is not None:
                (out / name).write_bytes(body)
