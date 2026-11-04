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

ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
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

    def read(self, offset: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)


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
                with urllib.request.urlopen(urllib.request.Request(target, headers=headers)) as r:
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


def parse_header(blob) -> tuple[dict[str, TensorRef], dict[str, str]]:
    if blob.size < 8:
        raise FormatError(f"{blob.name}: too short to be safetensors")
    (n,) = struct.unpack("<Q", blob.read(0, 8))
    if n > MAX_HEADER or 8 + n > blob.size:
        raise FormatError(f"{blob.name}: header length {n} is out of bounds")
    try:
        header = json.loads(blob.read(8, n))
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise FormatError(f"{blob.name}: header is not JSON") from e
    if not isinstance(header, dict):
        raise FormatError(f"{blob.name}: header is not an object")

    meta = header.pop("__metadata__", None) or {}
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


class Checkpoint:
    """Every tensor in a local directory, a single file, or a Hub repo at a revision."""

    def __init__(self, spec: str, tensors: dict[str, TensorRef], files: list[str], reader):
        self.spec = spec
        self.tensors = tensors
        self.files = files
        self._reader = reader

    def raw(self, path: str) -> bytes | None:
        return self._reader(path) if path in self.files else None

    def text(self, path: str) -> str | None:
        body = self.raw(path)
        return None if body is None else body.decode(errors="replace")

    @classmethod
    def open(cls, spec: str, token: str | None = None) -> Checkpoint:
        path = Path(spec).expanduser()
        if path.exists():
            return cls._local(path)
        return cls._hub(spec, token or os.environ.get("HF_TOKEN"))

    @classmethod
    def _local(cls, path: Path) -> Checkpoint:
        root = path if path.is_dir() else path.parent
        shards = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        if not shards:
            raise FormatError(f"{path}: no safetensors files")
        tensors = {}
        for shard in shards:
            tensors.update(parse_header(LocalBlob(shard))[0])
        files = sorted(
            str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts
        )
        return cls(str(path), tensors, files, lambda f: (root / f).read_bytes())

    @classmethod
    def _hub(cls, spec: str, token: str | None) -> Checkpoint:
        repo, _, rev = spec.partition("@")
        rev = rev or "main"
        req = urllib.request.Request(f"{HF}/api/models/{repo}/revision/{rev}")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as r:
            info = json.load(r)
        sha = info["sha"]
        files = sorted(s["rfilename"] for s in info["siblings"])
        shards = [f for f in files if f.endswith(".safetensors")]
        if not shards:
            raise FormatError(f"{spec}: no safetensors files in this repo")

        def url(f: str) -> str:
            return f"{HF}/{repo}/resolve/{sha}/{f}"

        def read(f: str) -> bytes:
            r = urllib.request.Request(url(f), headers={"User-Agent": "neyra"})
            with urllib.request.urlopen(r) as resp:
                return resp.read()

        tensors = {}
        for shard in shards:
            tensors.update(parse_header(HttpBlob(url(shard), shard, token))[0])
        return cls(f"{repo}@{sha}", tensors, files, read)
