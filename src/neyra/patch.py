"""A weight edit as a small, exact, replayable file.

A patch is itself a safetensors file, so reading one cannot execute code. Each
edited tensor records the hash it must be applied to and the hash it must produce.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .numerics import from_float, is_float, to_float
from .store import ITEMSIZE, Checkpoint, LocalBlob, parse_header

FORMAT = 1


class PatchMismatch(ValueError):
    pass


def bits_sha256(bits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


@dataclass
class TensorEdit:
    name: str
    dtype: str
    shape: tuple[int, ...]
    base_sha256: str
    edited_sha256: str | None
    idx: np.ndarray
    val: np.ndarray
    left: np.ndarray | None = None
    right: np.ndarray | None = None

    @property
    def low_rank(self) -> bool:
        return self.left is not None

    def apply(self, base_bits: np.ndarray, verify: bool = True) -> np.ndarray:
        if verify and bits_sha256(base_bits) != self.base_sha256:
            raise PatchMismatch(f"{self.name}: this is not the tensor the edit was made on")
        if self.low_rank:
            w = to_float(base_bits, self.dtype).astype(np.float64)
            for k in range(self.left.shape[1]):
                w = w + np.outer(self.left[:, k], self.right[k])
            out = from_float(w.astype(np.float32), self.dtype).reshape(-1)
        else:
            out = np.array(base_bits, copy=True).reshape(-1)
        out[self.idx] = self.val
        out = out.reshape(self.shape)
        if verify and self.edited_sha256 and bits_sha256(out) != self.edited_sha256:
            raise PatchMismatch(f"{self.name}: rebuilt tensor does not match the recorded hash")
        return out

    def dense(self, base_bits: np.ndarray) -> TensorEdit:
        """The same edit with every changed entry listed, so it can be split."""
        if not self.low_rank:
            return self
        edited = self.apply(base_bits, verify=False)
        idx = np.flatnonzero(base_bits.reshape(-1) != edited.reshape(-1))
        return TensorEdit(self.name, self.dtype, self.shape, self.base_sha256,
                          self.edited_sha256, idx, edited.reshape(-1)[idx])


@dataclass
class Patch:
    edits: list[TensorEdit]
    meta: dict = field(default_factory=dict)

    @property
    def entries(self) -> int:
        return sum(len(e.idx) for e in self.edits)

    def apply(self, ckpt: Checkpoint, verify: bool = True) -> dict[str, np.ndarray]:
        out = {}
        for e in self.edits:
            ref = ckpt.tensors.get(e.name)
            if ref is None:
                raise PatchMismatch(f"{e.name}: not in {ckpt.spec}")
            out[e.name] = e.apply(ref.read(), verify)
        return out

    def invert(self, ckpt: Checkpoint) -> Patch:
        inverse = []
        for e in self.edits:
            base = ckpt.tensors[e.name].read()
            full = e.dense(base)
            inverse.append(TensorEdit(e.name, e.dtype, e.shape, e.edited_sha256 or bits_sha256(e.apply(base)),
                                      e.base_sha256, full.idx, base.reshape(-1)[full.idx]))
        return Patch(inverse, {**self.meta, "inverse_of": self.fingerprint})

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for e in sorted(self.edits, key=lambda e: e.name):
            h.update(e.name.encode())
            h.update(e.base_sha256.encode())
            for arr in (e.idx, e.val, e.left, e.right):
                if arr is not None:
                    h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def save(self, path: str | Path) -> None:
        tensors: dict[str, np.ndarray] = {}
        manifest = []
        for i, e in enumerate(self.edits):
            tensors[f"{i}.idx"] = e.idx.astype(np.int64)
            tensors[f"{i}.val"] = e.val
            if e.low_rank:
                tensors[f"{i}.left"] = e.left
                tensors[f"{i}.right"] = e.right
            manifest.append({"name": e.name, "dtype": e.dtype, "shape": list(e.shape),
                             "base_sha256": e.base_sha256, "edited_sha256": e.edited_sha256})
        write_safetensors(path, tensors, {
            "neyra": json.dumps({"format": FORMAT, "tensors": manifest, "meta": self.meta})
        })

    @classmethod
    def load(cls, path: str | Path) -> Patch:
        refs, meta = parse_header(LocalBlob(Path(path)))
        if "neyra" not in meta:
            raise PatchMismatch(f"{path}: not a neyra patch")
        doc = json.loads(meta["neyra"])
        if doc["format"] != FORMAT:
            raise PatchMismatch(f"{path}: patch format {doc['format']} is not supported")

        def arr(key: str, kind=None) -> np.ndarray | None:
            ref = refs.get(key)
            if ref is None:
                return None
            return ref.read().view(kind) if kind else ref.read()

        edits = [
            TensorEdit(t["name"], t["dtype"], tuple(t["shape"]), t["base_sha256"], t["edited_sha256"],
                       arr(f"{i}.idx", np.int64), arr(f"{i}.val"),
                       arr(f"{i}.left", np.float64), arr(f"{i}.right", np.float64))
            for i, t in enumerate(doc["tensors"])
        ]
        return cls(edits, doc.get("meta", {}))


def diff_tensor(name: str, dtype: str, base: np.ndarray, edited: np.ndarray) -> TensorEdit:
    """Smallest exact encoding of edited relative to base."""
    idx = np.flatnonzero(base.reshape(-1) != edited.reshape(-1))
    sparse = TensorEdit(name, dtype, tuple(base.shape), bits_sha256(base), bits_sha256(edited),
                        idx, edited.reshape(-1)[idx])
    if base.ndim != 2 or not is_float(dtype) or len(idx) < 0.25 * base.size:
        return sparse

    b = to_float(base, dtype).astype(np.float64)
    d = to_float(edited, dtype).astype(np.float64) - b
    u, s, vh = np.linalg.svd(d, full_matrices=False)
    budget = len(idx) * (8 + ITEMSIZE[dtype])
    for rank in (1, 2, 4, 8):
        left, right = u[:, :rank] * s[:rank], vh[:rank]
        w = b
        for k in range(rank):
            w = w + np.outer(left[:, k], right[k])
        approx = from_float(w.astype(np.float32), dtype).reshape(-1)
        wrong = np.flatnonzero(approx != edited.reshape(-1))
        size = left.nbytes + right.nbytes + len(wrong) * (8 + ITEMSIZE[dtype])
        if size < budget:
            return TensorEdit(name, dtype, tuple(base.shape), sparse.base_sha256, sparse.edited_sha256,
                              wrong, edited.reshape(-1)[wrong], left, right)
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
    return Patch(edits, meta or {})


def write_safetensors(path: str | Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    names = {np.dtype(np.int64): "I64", np.dtype(np.int32): "I32", np.dtype(np.float64): "F64",
             np.dtype(np.float32): "F32", np.dtype(np.uint16): "BF16", np.dtype(np.uint32): "F32",
             np.dtype(np.uint64): "U64", np.dtype(np.uint8): "U8"}
    header: dict = {"__metadata__": metadata} if metadata else {}
    offset = 0
    for name, arr in tensors.items():
        header[name] = {"dtype": names[arr.dtype], "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
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
            header[r.name] = {"dtype": r.dtype, "shape": list(r.shape),
                              "data_offsets": [offset, offset + r.nbytes]}
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
