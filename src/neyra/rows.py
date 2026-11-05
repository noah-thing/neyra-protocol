"""Row provable roots, Annex B of the protocol.

Every tensor gets a Merkle tree over groups of whole rows, and the trees compose into one root.
A few rows of a remote checkpoint can then be checked against that root by reading those rows
and a proof of a few hundred bytes. The weights root is untouched. The two are tied by a link
that someone who read the whole model signs, and that tie is the one thing a proof cannot carry.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .patch import write_safetensors
from .protocol import (
    Opened,
    ProtocolError,
    Table,
    check_signature,
    decode_envelope,
    envelope,
    jcs,
    root_of,
    strict_loads,
    tensor_prefix,
)
from .store import (
    CHUNK,
    ITEMSIZE,
    RAW_DTYPE,
    Checkpoint,
    FormatError,
    LocalBlob,
    TensorRef,
    parse_header,
)

INDEX_TYPE = "neyra.rows/1"
INDEX_KEY = "neyra.rows"
PROOF_TYPE = "neyra.rowproof/1"
LINK_PREDICATE = "https://neyra.run/rows/v1"
LEAF_BYTES = 4096
HASH = 32
PROOF_MEMBERS = {"type", "rows_root", "tensor", "index", "count", "ranges", "tree", "path"}


def _h(domain: str, data: bytes) -> bytes:
    return hashlib.sha256(f"neyra/{domain}/v1\0".encode() + data).digest()


def text(h: bytes) -> str:
    return "sha256:" + h.hex()


def _raw(d: str) -> bytes:
    return bytes.fromhex(d[7:])


def _split(n: int) -> int:
    """The largest power of two below n, which is where RFC 6962 cuts a tree."""
    return 1 << ((n - 1).bit_length() - 1)


def tree_root(buf: bytes, lo: int = 0, hi: int | None = None) -> bytes:
    """Root over the hashes lo to hi of a buffer of 32 byte hashes."""
    hi = len(buf) // HASH if hi is None else hi
    n = hi - lo
    if n == 0:
        return _h("rownode", b"")
    if n == 1:
        return buf[lo * HASH:hi * HASH]
    mid = lo + _split(n)
    return _h("rownode", tree_root(buf, lo, mid) + tree_root(buf, mid, hi))


def open_leaves(buf: bytes, wanted: list[int]) -> list[bytes]:
    """The subtree hashes that, with the wanted leaves, rebuild the root, left to right."""
    out: list[bytes] = []

    def walk(lo: int, hi: int) -> None:
        i = bisect_left(wanted, lo)
        if i == len(wanted) or wanted[i] >= hi:
            out.append(tree_root(buf, lo, hi))
        elif hi - lo > 1:
            mid = lo + _split(hi - lo)
            walk(lo, mid)
            walk(mid, hi)

    walk(0, len(buf) // HASH)
    return out


def close_leaves(n: int, known: dict[int, bytes], proof: list[bytes]) -> bytes:
    """Rebuild the root of n leaves from some of them and the hashes open_leaves gave."""
    wanted = sorted(known)
    spare = iter(proof)

    def take() -> bytes:
        try:
            return next(spare)
        except StopIteration:
            raise ProtocolError("bad_proof", "the proof has too few hashes") from None

    def walk(lo: int, hi: int) -> bytes:
        i = bisect_left(wanted, lo)
        if i == len(wanted) or wanted[i] >= hi:
            return take()
        if hi - lo == 1:
            return known[lo]
        mid = lo + _split(hi - lo)
        return _h("rownode", walk(lo, mid) + walk(mid, hi))

    root = walk(0, n) if n else _h("rownode", b"")
    if next(spare, None) is not None:
        raise ProtocolError("bad_proof", "the proof has too many hashes")
    return root


@dataclass(frozen=True)
class Geometry:
    rows: int
    width: int  # bytes in one row
    group: int  # rows in one leaf, 0 for a tensor with no bytes

    @classmethod
    def of(cls, dtype: str, shape: tuple[int, ...]) -> Geometry:
        rows = shape[0] if shape else 1
        width = math.prod(shape[1:]) * ITEMSIZE[dtype]
        return cls(rows, width, 0 if rows * width == 0 else -(-LEAF_BYTES // width))

    @property
    def leaves(self) -> int:
        return -(-self.rows // self.group) if self.group else 0

    def span(self, leaf: int) -> tuple[int, int]:
        """Byte range of a leaf inside the tensor."""
        return leaf * self.group * self.width, min(self.rows, (leaf + 1) * self.group) * self.width

    def covering(self, ranges: list[tuple[int, int]]) -> list[int]:
        return sorted({k for a, b in ranges for k in range(a // self.group, (b - 1) // self.group + 1)})


def tensor_leaf(name: str, dtype: str, shape, tree: bytes) -> bytes:
    return _h("rowtensor", jcs({"name": name, "dtype": dtype, "shape": list(shape), "tree": text(tree)}))


def parse_ranges(spec: str) -> list[tuple[int, int]]:
    """'0:10,50:60' as sorted ranges, with overlapping or touching ones merged."""
    out: list[list[int]] = []
    try:
        parts = sorted(tuple(int(x) for x in part.split(":")) for part in spec.split(","))
        if any(len(p) != 2 for p in parts):
            raise ValueError
    except ValueError:
        raise FormatError(f"{spec!r}: rows are written start:stop, such as 0:10 or 0:10,50:60") from None
    for a, b in parts:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


@dataclass
class Index:
    """Per tensor leaf hashes, plus the flat digests that make up the weights root."""

    entries: list[dict]  # name, dtype, shape, digest, tree; sorted by name
    leaves: dict[str, bytes]

    @property
    def weights_root(self) -> str:
        return root_of({e["name"]: {k: e[k] for k in ("name", "dtype", "shape", "digest")} for e in self.entries})

    @property
    def rows_root(self) -> str:
        return text(tree_root(self._tensor_leaves()))

    def _tensor_leaves(self) -> bytes:
        return b"".join(tensor_leaf(e["name"], e["dtype"], e["shape"], _raw(e["tree"])) for e in self.entries)

    def entry(self, name: str) -> tuple[int, dict]:
        for i, e in enumerate(self.entries):
            if e["name"] == name:
                return i, e
        raise KeyError(name)

    def manifest(self) -> dict:
        return {"type": INDEX_TYPE, "weights_root": self.weights_root, "rows_root": self.rows_root,
                "tensors": self.entries}

    def save(self, path: str | Path) -> None:
        payload = {str(i): np.frombuffer(self.leaves[e["name"]], dtype=np.uint8).reshape(-1, HASH)
                   for i, e in enumerate(self.entries) if self.leaves[e["name"]]}
        write_safetensors(path, payload, {INDEX_KEY: jcs(self.manifest()).decode()})

    @classmethod
    def load(cls, path: str | Path) -> Index:
        refs, meta = parse_header(LocalBlob(Path(path)))
        doc = strict_loads(meta.get(INDEX_KEY, "null"))
        if not isinstance(doc, dict) or doc.get("type") != INDEX_TYPE or not isinstance(doc.get("tensors"), list):
            raise ProtocolError("bad_manifest", "not a row index")
        leaves = {}
        for i, e in enumerate(doc["tensors"]):
            geo = Geometry.of(e["dtype"], tuple(e["shape"]))
            ref = refs.get(str(i))
            buf = ref.read().tobytes() if ref is not None else b""
            if len(buf) != geo.leaves * HASH or text(tree_root(buf)) != e["tree"]:
                raise ProtocolError("bad_manifest", f"{e['name']}: leaves do not match their tree")
            leaves[e["name"]] = buf
        index = cls(doc["tensors"], leaves)
        if doc.get("weights_root") != index.weights_root or doc.get("rows_root") != index.rows_root:
            raise ProtocolError("bad_manifest", "roots do not match the tensors")
        return index


def hash_tensor(ref: TensorRef) -> tuple[str, bytes]:
    """Flat digest and leaf hashes in one read."""
    geo = Geometry.of(ref.dtype, ref.shape)
    flat = hashlib.sha256(tensor_prefix(ref.dtype, ref.shape))
    leaves = bytearray()
    step = max(1, CHUNK // (geo.group * geo.width)) if geo.group else 0
    for first in range(0, geo.leaves, step or 1):
        last = min(first + step, geo.leaves)
        lo, hi = geo.span(first)[0], geo.span(last - 1)[1]
        data = ref.blob.read(ref.start + lo, hi - lo)
        flat.update(data)
        for k in range(first, last):
            a, b = geo.span(k)
            leaves += _h("rowleaf", data[a - lo:b - lo])
    return text(flat.digest()), bytes(leaves)


def build_index(ck: Checkpoint) -> Index:
    entries, leaves = [], {}
    for name in sorted(ck.tensors, key=lambda n: n.encode("utf-16-be", "surrogatepass")):
        ref = ck.tensors[name]
        flat, buf = hash_tensor(ref)
        entries.append({"name": name, "dtype": ref.dtype, "shape": list(ref.shape), "digest": flat,
                        "tree": text(tree_root(buf))})
        leaves[name] = buf
    return Index(entries, leaves)


def check_index(index: Index, table: Table) -> None:
    """Whether an index agrees with a weights table the verifier already trusts."""
    if index.weights_root != root_of(table):
        raise ProtocolError("rows_mismatch", "the index describes different weights")


def prove(index: Index, name: str, ranges: list[tuple[int, int]]) -> dict:
    i, e = index.entry(name)
    geo = Geometry.of(e["dtype"], tuple(e["shape"]))
    _check_ranges(ranges, geo)
    return {
        "type": PROOF_TYPE,
        "rows_root": index.rows_root,
        "tensor": {"name": name, "dtype": e["dtype"], "shape": e["shape"]},
        "index": i,
        "count": len(index.entries),
        "ranges": [list(r) for r in ranges],
        "tree": [text(h) for h in open_leaves(index.leaves[name], geo.covering(ranges))],
        "path": [text(h) for h in open_leaves(index._tensor_leaves(), [i])],
    }


def _check_ranges(ranges, geo: Geometry) -> None:
    if not geo.group:
        raise ProtocolError("bad_proof", "the tensor has no bytes to prove")
    cursor = 0
    if not ranges:
        raise ProtocolError("bad_proof", "no rows asked for")
    for a, b in ranges:
        if not (cursor <= a < b <= geo.rows):
            raise ProtocolError("bad_proof", "ranges must be non empty, sorted, disjoint and inside the tensor")
        cursor = b


def opened(proof: dict) -> list[tuple[int, int]]:
    """Byte ranges of the tensor to read, one per run of adjacent leaves."""
    tensor = proof["tensor"]
    geo = Geometry.of(tensor["dtype"], tuple(tensor["shape"]))
    _check_ranges(proof["ranges"], geo)
    runs: list[list[int]] = []
    for k in geo.covering([tuple(r) for r in proof["ranges"]]):
        a, b = geo.span(k)
        if runs and runs[-1][1] == a:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return [(a, b) for a, b in runs]


def fetch(ref: TensorRef, proof: dict) -> bytes:
    return b"".join(ref.blob.read(ref.start + a, b - a) for a, b in opened(proof))


def _digests(value, what: str) -> list[bytes]:
    if not isinstance(value, list) or not all(
            isinstance(d, str) and len(d) == 71 and d.startswith("sha256:") and _is_hex(d[7:]) for d in value):
        raise ProtocolError("bad_proof", f"{what} must be a list of digests")
    return [_raw(d) for d in value]


def _is_hex(s: str) -> bool:
    return all(c in "0123456789abcdef" for c in s)


def _count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def verify_proof(proof, data: bytes) -> list[bytes]:
    """Check rows against a proof, and return the rows asked for, one block per range.

    data is the bytes opened() names, in order. The proof says what the rows hash to, and not
    that the root belongs to any particular model. That is the link's job.
    """
    bad = "bad_proof"
    if not isinstance(proof, dict) or set(proof) != PROOF_MEMBERS or proof["type"] != PROOF_TYPE:
        raise ProtocolError(bad, "not a row proof")
    tensor, ranges = proof["tensor"], proof["ranges"]
    if (not isinstance(tensor, dict) or set(tensor) != {"name", "dtype", "shape"}
            or not isinstance(tensor["name"], str) or not isinstance(tensor["dtype"], str) or tensor["dtype"] not in ITEMSIZE
            or not isinstance(tensor["shape"], list) or not all(map(_count, tensor["shape"]))):
        raise ProtocolError(bad, "malformed tensor")
    if not (_count(proof["index"]) and _count(proof["count"]) and proof["index"] < proof["count"]):
        raise ProtocolError(bad, "malformed position")
    if (not isinstance(ranges, list) or not all(isinstance(r, list) and len(r) == 2 and all(map(_count, r))
                                                  for r in ranges)):
        raise ProtocolError(bad, "malformed ranges")
    root = _digests([proof["rows_root"]], "rows_root")[0]
    tree, path = _digests(proof["tree"], "tree"), _digests(proof["path"], "path")
    shape = tuple(tensor["shape"])
    geo = Geometry.of(tensor["dtype"], shape)
    ranges = [tuple(r) for r in ranges]
    _check_ranges(ranges, geo)
    if sum(b - a for a, b in ranges) * geo.width > len(data):  # before a forged range can cost memory
        raise ProtocolError(bad, "data is shorter than the rows asked for")

    leaves = geo.covering(ranges)
    spans = [geo.span(k) for k in leaves]
    if len(data) != sum(b - a for a, b in spans):
        raise ProtocolError(bad, "data is not the bytes of the opened leaves")
    known, cursor = {}, 0
    for k, (a, b) in zip(leaves, spans):
        known[k] = _h("rowleaf", data[cursor:cursor + b - a])
        cursor += b - a
    top = close_leaves(geo.leaves, known, tree)
    leaf = tensor_leaf(tensor["name"], tensor["dtype"], shape, top)
    if close_leaves(proof["count"], {proof["index"]: leaf}, path) != root:
        raise ProtocolError("proof_mismatch", "these rows do not belong to this root")

    where = {k: o for k, o in zip(leaves, _offsets(spans))}
    out = []
    for a, b in ranges:
        k = a // geo.group
        start = where[k] + (a - k * geo.group) * geo.width
        out.append(data[start:start + (b - a) * geo.width])
    return out


def _offsets(spans: list[tuple[int, int]]) -> list[int]:
    out, at = [], 0
    for a, b in spans:
        out.append(at)
        at += b - a
    return out


def rows_array(dtype: str, shape: tuple[int, ...], block: bytes) -> np.ndarray:
    """Raw bits of a block of rows, shaped, with floats left as unsigned ints."""
    arr = np.frombuffer(block, dtype=RAW_DTYPE[ITEMSIZE[dtype]])
    return arr.reshape(-1, *shape[1:]) if shape else arr


def link_statement(index: Index) -> dict:
    weights, rows = index.weights_root, index.rows_root
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": "weights", "digest": {"sha256": weights[7:]}},
                    {"name": "rows", "digest": {"sha256": rows[7:]}}],
        "predicateType": LINK_PREDICATE,
        "predicate": {"digest_algorithm": "neyra/rows/v1", "weights_root": weights, "rows_root": rows,
                      "tensors": len(index.entries)},
    }


def link(index: Index, key: bytes | None = None) -> dict:
    return envelope(link_statement(index), key)


def open_link(doc: dict, public: bytes | None = None) -> Opened:
    statement, payload = decode_envelope(doc)
    predicate = statement.get("predicate")
    if statement.get("_type") != "https://in-toto.io/Statement/v1" or statement.get("predicateType") != LINK_PREDICATE:
        raise ProtocolError("bad_type", "not a row link")
    if (not isinstance(predicate, dict) or set(predicate) != {"digest_algorithm", "weights_root", "rows_root", "tensors"}
            or predicate["digest_algorithm"] != "neyra/rows/v1"):
        raise ProtocolError("bad_claim", "malformed link")
    want = [{"name": "weights", "digest": {"sha256": str(predicate["weights_root"])[7:]}},
            {"name": "rows", "digest": {"sha256": str(predicate["rows_root"])[7:]}}]
    if statement.get("subject") != want:
        raise ProtocolError("subject_mismatch", "subject does not match the link")
    if public is not None:
        check_signature(doc, payload, public)
    return Opened(statement, public is not None)

