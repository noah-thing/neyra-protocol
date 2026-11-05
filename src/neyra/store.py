"""Read safetensors checkpoints from disk or over HTTP range requests.

Nothing here needs torch. Tensors come back as numpy arrays of their raw bits
(bfloat16 as uint16), so hashes and patches are exact by construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HF = "https://huggingface.co"
MAX_HEADER = 100 * 1024 * 1024
CHUNK = 16 * 1024 * 1024
TIMEOUT = 60

ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}
RAW_DTYPE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


class FormatError(ValueError):
    pass


class LocalBlob:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.name = self.path.name
        self.bytes_read = 0

    def read(self, offset: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(offset)
            data = f.read(length)
        self.bytes_read += len(data)
        return data


class HttpBlob:
    """A remote file read in ranges. Signed CDN redirects are refreshed on expiry."""

    def __init__(self, url: str, name: str, token: str | None = None):
        self.url = url
        self.name = name
        self.token = token
        self.direct: str | None = None
        self.bytes_read = 0
        self.size = int(self._fetch(0, 1)[1].split("/")[-1])

    def _fetch(self, offset: int, length: int) -> tuple[bytes, str]:
        last: Exception | None = None
        for attempt in range(5):
            headers = {"Range": f"bytes={offset}-{offset + length - 1}", "User-Agent": "neyra"}
            target = self.direct or self.url
            if self.token and self.direct is None:
                headers["Authorization"] = f"Bearer {self.token}"
            try:
                with urllib.request.urlopen(urllib.request.Request(target, headers=headers), timeout=TIMEOUT) as r:
                    if r.status != 206:
                        raise FormatError(f"{self.name}: server ignored the range request")
                    self.direct = r.geturl()
                    return r.read(), r.headers["Content-Range"]
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (401, 404):
                    raise
                self.direct = None
            except OSError as e:
                last = e
            time.sleep(2**attempt)
        raise last

    def read(self, offset: int, length: int) -> bytes:
        data = self._fetch(offset, length)[0]
        self.bytes_read += len(data)
        return data


@dataclass(frozen=True)
class TensorRef:
    name: str
    dtype: str
    shape: tuple[int, ...]
    blob: LocalBlob | HttpBlob
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def read(self) -> np.ndarray:
        """Raw bits, shaped. Floats stay as unsigned ints of the same width."""
        parts = [
            self.blob.read(o, min(CHUNK, self.end - o))
            for o in range(self.start, self.end, CHUNK)
        ]
        raw = np.frombuffer(b"".join(parts), dtype=RAW_DTYPE[ITEMSIZE[self.dtype]])
        return raw.reshape(self.shape)

    def read_slice(self, first: int, count: int) -> np.ndarray:
        """Raw bits of `count` entries from flat entry `first`, without reading the rest of the tensor."""
        size = ITEMSIZE[self.dtype]
        raw = self.blob.read(self.start + first * size, count * size)
        return np.frombuffer(raw, dtype=RAW_DTYPE[size])

    def sha256(self) -> str:
        h = hashlib.sha256()
        for o in range(self.start, self.end, CHUNK):
            h.update(self.blob.read(o, min(CHUNK, self.end - o)))
        return h.hexdigest()


def _is_count(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _entry(file: str, name: str, info) -> tuple[str, tuple[int, ...], int, int]:
    if not isinstance(info, dict):
        raise FormatError(f"{file}: {name} is not a tensor entry")
    dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
    if not isinstance(dtype, str) or dtype not in ITEMSIZE:
        raise FormatError(f"{file}: {name} has unsupported dtype {dtype!r}")
    if not isinstance(shape, list) or not all(_is_count(d) for d in shape):
        raise FormatError(f"{file}: {name} has an invalid shape")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(_is_count(o) for o in offsets):
        raise FormatError(f"{file}: {name} has invalid offsets")
    begin, end = offsets
    expected = math.prod(shape) * ITEMSIZE[dtype]
    if end - begin != expected:
        raise FormatError(f"{file}: {name} spans {end - begin} bytes, its shape needs {expected}")
    return dtype, tuple(shape), begin, end


def unique_keys(pairs):
    """Two readers must not see different tensors under one name, so a repeated key is an error, not last-wins."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def parse_header(blob) -> tuple[dict[str, TensorRef], dict[str, str]]:
    if blob.size < 8:
        raise FormatError(f"{blob.name}: too short to be safetensors")
    (n,) = struct.unpack("<Q", blob.read(0, 8))
    if n > MAX_HEADER or 8 + n > blob.size:
        raise FormatError(f"{blob.name}: header length {n} is out of bounds")
    try:
        header = json.loads(blob.read(8, n), object_pairs_hook=unique_keys)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise FormatError(f"{blob.name}: header is not JSON") from e
    if not isinstance(header, dict):
        raise FormatError(f"{blob.name}: header is not an object")

    meta = header.pop("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(v, str) for v in meta.values()):
        raise FormatError(f"{blob.name}: __metadata__ must map strings to strings")
    base = 8 + n
    refs = {}
    spans = []
    for name, info in header.items():
        dtype, shape, begin, end = _entry(blob.name, name, info)
        spans.append((begin, end, name))
        refs[name] = TensorRef(name, dtype, shape, blob, base + begin, base + end)

    # Overlapping or gapped offsets are how a crafted file hides or aliases data.
    cursor = 0
    for begin, end, name in sorted(spans):
        if begin != cursor:
            raise FormatError(f"{blob.name}: {name} does not start where the last tensor ended")
        cursor = end
    if base + cursor != blob.size:
        raise FormatError(f"{blob.name}: {blob.size - base - cursor} trailing bytes after tensors")
    return refs, meta


def merge(shards) -> dict[str, TensorRef]:
    tensors: dict[str, TensorRef] = {}
    for refs in shards:
        clash = tensors.keys() & refs.keys()
        if clash:
            raise FormatError(f"tensor {min(clash)} appears in two files")
        tensors.update(refs)
    return tensors


def weight_files(files: list[str]) -> list[str]:
    """The top level weight files, which is what PROTOCOL.md section 3 calls a model: dot files are skipped."""
    return [f for f in files if f.endswith(".safetensors") and "/" not in f and not f.startswith(".")]


def canonical_shards(files: list[str]) -> list[str]:
    """The Hugging Face weights of a repo, skipping second copies such as original/ or consolidated."""
    top = weight_files(files)
    return [f for f in top if not f.startswith("consolidated")] or top


class Checkpoint:
    """Every tensor in a local directory, a single file, or a Hub repo at a revision."""

    def __init__(self, spec: str, tensors: dict[str, TensorRef], files: list[str], reader, digests=None):
        self.spec = spec
        self.tensors = tensors
        self.files = files
        self._reader = reader
        self._digests = digests or {}
        self._root: Path | None = None

    def digest(self, path: str) -> str | None:
        """SHA-256 of a file. The Hub states it for large files without a download."""
        if path in self._digests:
            return self._digests[path]
        if self._root is not None and path in self.files:
            h = hashlib.sha256()
            with open(self._root / path, "rb") as f:
                while block := f.read(CHUNK):
                    h.update(block)
            found = h.hexdigest()
        else:
            body = self.raw(path)
            found = None if body is None else hashlib.sha256(body).hexdigest()
        self._digests[path] = found
        return found

    @property
    def size(self) -> int:
        return sum({id(r.blob): r.blob.size for r in self.tensors.values()}.values())

    @property
    def bytes_read(self) -> int:
        return sum({id(r.blob): getattr(r.blob, "bytes_read", 0) for r in self.tensors.values()}.values())

    def raw(self, path: str) -> bytes | None:
        return self._reader(path) if path in self.files else None

    def text(self, path: str) -> str | None:
        body = self.raw(path)
        return None if body is None else body.decode(errors="replace")

    @classmethod
    def open(cls, spec: str, token: str | None = None, protocol: bool = False) -> Checkpoint:
        """protocol=True reads a folder exactly as the spec defines a model, with no Hub convention applied."""
        path = Path(spec).expanduser()
        if path.exists():
            return cls._local(path, protocol)
        return cls._hub(spec, token or os.environ.get("HF_TOKEN"))

    @classmethod
    def _local(cls, path: Path, protocol: bool = False) -> Checkpoint:
        root = path if path.is_dir() else path.parent
        pick = weight_files if protocol else canonical_shards
        shards = [root / f for f in pick([p.name for p in path.iterdir() if p.is_file()])] if path.is_dir() else [path]
        if not shards:
            raise FormatError(f"{path}: no safetensors files")
        tensors = merge(parse_header(LocalBlob(shard))[0] for shard in shards)
        names = (p.relative_to(root) for p in root.rglob("*") if p.is_file())
        files = sorted(n.as_posix() for n in names if ".cache" not in n.parts)
        ck = cls(str(path), tensors, files, lambda f: (root / f).read_bytes())
        ck._root = root
        return ck

    @classmethod
    def _hub(cls, spec: str, token: str | None) -> Checkpoint:
        repo, _, rev = spec.partition("@")
        rev = rev or "main"
        req = urllib.request.Request(f"{HF}/api/models/{repo}/revision/{rev}?blobs=true")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            info = json.load(r)
        sha = info["sha"]
        files = sorted(s["rfilename"] for s in info["siblings"])
        shards = canonical_shards(files)
        if not shards:
            raise FormatError(f"{spec}: no safetensors files in this repo")

        def url(f: str) -> str:
            return f"{HF}/{repo}/resolve/{sha}/{f}"

        def read(f: str) -> bytes:
            r = urllib.request.Request(url(f), headers={"User-Agent": "neyra"})
            with urllib.request.urlopen(r, timeout=TIMEOUT) as resp:
                return resp.read()

        tensors = merge(parse_header(HttpBlob(url(shard), shard, token))[0] for shard in shards)
        digests = {s["rfilename"]: s["lfs"]["sha256"] for s in info["siblings"] if s.get("lfs")}
        return cls(f"{repo}@{sha}", tensors, files, read, digests)
