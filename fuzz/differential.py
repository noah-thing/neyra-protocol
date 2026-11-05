"""Differential fuzzer for Neyra Protocol verifiers.

Mutates valid protocol objects, runs the Python verifier (neyra.conformance) and the Go one (impl/go) on every
mutant, and reports each case where they answer differently. A disagreement is a bug in one of them or a place
the spec is silent; either way it belongs in spec/vectors.

    PYTHONPATH=src python fuzz/differential.py --cases 20000 --seed 1 --out findings/

Wrongly accepted input is ranked first: one verifier says `ok` (or returns a digest) and the other rejects.
Then answers that differ in value, then crashes, then rejections that differ only in the error code.
Cases the spec lets a verifier decline (the optional low-rank/1 extension) are counted apart.

Nothing here shares code with either verifier except as a source of valid seeds: mutation, JSON writing and
the comparison are written against the spec text only.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GO_DIR = REPO / "impl" / "go"
VECTORS = REPO / "spec" / "vectors"

# Rank for the report: lower is worse.
RANKS = ["accepted_by_one", "value_differs", "crash", "code_differs"]


# ---------------------------------------------------------------- writing cases to disk

class Bench:
    """A folder holding cases.json and the files those cases point at, in the vectors layout."""

    def __init__(self, root: Path):
        self.root = root
        self.cases: list[dict] = []
        self.files = 0
        root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, ext: str = "") -> str:
        self.files += 1
        name = f"f{self.files}{ext}"
        (self.root / name).write_bytes(data)
        return name

    def share(self, source: Path) -> str:
        """Copy a seed file or folder in once and return its path relative to the bench, which both verifiers can open."""
        target = self.root / "shared" / source.parent.name / source.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy(source, target)
        return str(target.relative_to(self.root))

    def folder(self) -> Path:
        self.files += 1
        path = self.root / f"d{self.files}"
        path.mkdir()
        return path

    def add(self, kind: str, label: str, **fields) -> None:
        self.cases.append({"kind": kind, "id": str(len(self.cases)), "label": label, **fields})

    def save(self) -> None:
        (self.root / "cases.json").write_text(json.dumps(self.cases))


# ---------------------------------------------------------------- JSON written by hand

WHITESPACE = [" ", "\t", "\n", "\r", "\f", "\v", "\u00a0", "\ufeff", "\u2028"]
NUMBER_TOKENS = ["0", "-0", "1", "-1", "9007199254740991", "9007199254740992", "-9007199254740991",
                 "-9007199254740992", "1.0", "1e2", "1E2", "0.5", "-", "+1", "01", "-01", "00", "1.", ".5", "0e0",
                 "1e-0", "123456789012345678901234567890", "-0.0", "0x10", "Infinity", "-Infinity", "NaN", "1_000",
                 "18446744073709551615", "18446744073709551616", "4294967296", "255", "256", "-2", "2.0"]
STRING_TOKENS = ['"a"', '"\\u0041"', '"\\ud83d\\ude00"', '"\\ud83d"', '"\\ude00"', '"\\ude00\\ud83d"', '"\\u0000"',
                 '"\\/"', '"\\x41"', "\"\\'\"", '"\t"', '"\x7f"', '"\u2028"', '"é"', '"\\U0041"', '"\\u00e9"',
                 '"\\uD83D\\uDE00"', '"\\u00zz"', '"\\"', '"\\n"', '"\n"', '"\x01"', '"\\u001f"', '"\\u007f"',
                 '"e\u0301"', '"\u00e9"', '"\U0001f600"', '"\uffff"', '"\\uffff"', '"\\ufb33"', '""']
KEY_POOL = ["a", "b", "aa", "A", "é", "\ufb33", "\U0001f600", "\uffff", "", "\x00", "\x1f", "\x7f", '"', "\\", "z",
            "\u2028", "e\u0301", "\U00010000", "\ud7ff", "\ue000"]


def esc(s: str) -> str:
    """Escape what RFC 8785 escapes and nothing else. Lone surrogates fall back to \\u escapes."""
    out = []
    for c in s:
        o = ord(c)
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif c in "\b\t\n\f\r":
            out.append({"\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}[c])
        elif o < 0x20 or 0xD800 <= o <= 0xDFFF:
            out.append(f"\\u{o:04x}")
        else:
            out.append(c)
    return '"' + "".join(out) + '"'


def utf16_key(s: str) -> bytes:
    return s.encode("utf-16-be", "surrogatepass")


def jcs(value) -> str:
    """Canonical JSON for the integers, strings, lists, dicts, bools and null that protocol objects hold."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return esc(value)
    if isinstance(value, float):
        return repr(value)  # not canonical on purpose: a float must be rejected wherever it lands
    if isinstance(value, list):
        return "[" + ",".join(jcs(v) for v in value) + "]"
    return "{" + ",".join(f"{esc(k)}:{jcs(value[k])}" for k in sorted(value, key=utf16_key)) + "}"


class Loose:
    """JSON written with chosen untidiness: key order, spacing, duplicate keys, odd number and string forms."""

    def __init__(self, rng: random.Random, shuffle=0.0, space=0.0, dup=0.0):
        self.rng, self.shuffle, self.space, self.dup = rng, shuffle, space, dup

    def ws(self) -> str:
        return self.rng.choice(WHITESPACE[:4]) if self.rng.random() < self.space else ""

    def dumps(self, v) -> str:
        r = self.rng
        if isinstance(v, dict):
            items = list(v.items())
            if self.shuffle and r.random() < self.shuffle:
                r.shuffle(items)
            if self.dup and items and r.random() < self.dup:
                k, val = r.choice(items)
                items.insert(r.randrange(len(items) + 1), (k, r.choice([val, 0, "x", None, [], {}])))
            body = ("," + self.ws()).join(f"{esc(k)}{self.ws()}:{self.ws()}{self.dumps(x)}" for k, x in items)
            return "{" + self.ws() + body + self.ws() + "}"
        if isinstance(v, list):
            return "[" + self.ws() + ("," + self.ws()).join(self.dumps(x) for x in v) + self.ws() + "]"
        return jcs(v)


# ---------------------------------------------------------------- mutation

def flip_bytes(rng: random.Random, b: bytes) -> tuple[bytes, str]:
    if not b:
        return bytes(rng.randrange(256) for _ in range(rng.randrange(1, 4))), "grow-empty"
    b = bytearray(b)
    op = rng.choice(["bit", "byte", "delete", "insert", "duplicate", "truncate", "swap", "zero-run", "ff-run"])
    i = rng.randrange(len(b))
    if op == "bit":
        b[i] ^= 1 << rng.randrange(8)
    elif op == "byte":
        b[i] = rng.randrange(256)
    elif op == "delete":
        del b[i:i + rng.randrange(1, 5)]
    elif op == "insert":
        b[i:i] = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 4)))
    elif op == "duplicate":
        j = min(len(b), i + rng.randrange(1, 12))
        b[i:i] = b[i:j]
    elif op == "truncate":
        del b[i:]
    elif op == "swap" and len(b) > 1:
        j = rng.randrange(len(b))
        b[i], b[j] = b[j], b[i]
    elif op == "zero-run":
        b[i:i + 4] = b"\x00" * len(b[i:i + 4])
    elif op == "ff-run":
        b[i:i + 4] = b"\xff" * len(b[i:i + 4])
    return bytes(b), f"bytes:{op}"


STRING_EDITS = ["upper", "append-nul", "append-newline", "prepend-space", "astral", "nfd", "truncate", "double",
                "empty", "trailing-space", "swap-case-hex", "fullwidth-digits"]


def edit_string(rng: random.Random, s: str) -> tuple[str, str]:
    op = rng.choice(STRING_EDITS)
    out = {
        "upper": s.upper(), "append-nul": s + "\x00", "append-newline": s + "\n", "prepend-space": " " + s,
        "astral": s + "\U0001f600", "nfd": __import__("unicodedata").normalize("NFD", s + "é"),
        "truncate": s[:-1], "double": s + s, "empty": "", "trailing-space": s + " ",
        "swap-case-hex": s.replace("a", "A").replace("f", "F"),
        "fullwidth-digits": s.translate({ord(str(d)): ord("０") + d for d in range(10)}),
    }[op]
    return out, f"str:{op}"


ATOMS = [None, True, False, 0, 1, -1, 2, 255, 2**31, 2**32, 2**53 - 1, 2**53, 2**63, 2**64, -(2**53), "", "x", [], {},
         [0], ["a"], {"a": 1}, 1.5, "sha256:" + "0" * 64, "sha256:" + "A" * 64, "sha256:" + "0" * 63, "ok"]


def paths(value, prefix=()):
    yield prefix
    if isinstance(value, dict):
        for k, v in value.items():
            yield from paths(v, prefix + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from paths(v, prefix + (i,))


def at(root, path):
    for p in path:
        root = root[p]
    return root


def clone(v):
    return json.loads(json.dumps(v))


def mutate_json(rng: random.Random, doc, atoms=ATOMS) -> tuple[object, str]:
    """One structural edit somewhere in a JSON document. Returns the new document and what was done."""
    doc = clone(doc)
    path = rng.choice(list(paths(doc)))
    if not path:
        return rng.choice(atoms), "root:replace"
    parent, key = at(doc, path[:-1]), path[-1]
    node = parent[key]
    op = rng.choice(["replace", "replace", "delete", "string", "dup", "type", "wrap", "swap", "pop", "add"])
    where = "/".join(map(str, path))
    if op == "delete":
        if isinstance(parent, dict):
            del parent[key]
        else:
            parent.pop(key)
    elif op == "string" and isinstance(node, str):
        parent[key], what = edit_string(rng, node)
        return doc, f"{where}:{what}"
    elif op == "dup" and isinstance(parent, list):
        parent.insert(key, clone(node))
    elif op == "wrap":
        parent[key] = rng.choice([[node], {"a": node}])
    elif op == "swap" and isinstance(parent, list) and len(parent) > 1:
        j = rng.randrange(len(parent))
        parent[key], parent[j] = parent[j], parent[key]
    elif op == "pop" and isinstance(node, list) and node:
        node.pop(rng.randrange(len(node)))
    elif op == "add" and isinstance(node, dict):
        node[rng.choice(["x-note", "surprise", "", "type", "A"])] = rng.choice(atoms)
    elif op == "type" and isinstance(node, int) and not isinstance(node, bool):
        parent[key] = rng.choice([node + 1, node - 1, str(node), [node], bool(node)])
    else:
        parent[key] = rng.choice(atoms)
        op = "replace"
    return doc, f"{where}:{op}"


# ---------------------------------------------------------------- independent protocol helpers

def domain_digest(domain: str, data: bytes) -> str:
    return "sha256:" + hashlib.sha256(b"neyra/" + domain.encode() + b"/v1\x00" + data).hexdigest()


def pae(payload_type: str, payload: bytes) -> bytes:
    t = payload_type.encode()
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def sign(secret: bytes, payload_type: str, payload: bytes) -> tuple[str, str]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    key = Ed25519PrivateKey.from_private_bytes(secret)
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return domain_digest("key", public), b64(key.sign(pae(payload_type, payload)))


def public_hex(secret: bytes) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return Ed25519PrivateKey.from_private_bytes(secret).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def safetensors(header: dict, data: bytes, pad: bytes = b"", raw_header: bytes | None = None, length: int | None = None):
    """A file from a header dict (or raw header bytes) and tensor data, padded to 8 bytes."""
    text = raw_header if raw_header is not None else json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    text += pad
    text += b" " * (-len(text) % 8) if not pad else b""
    return struct.pack("<Q", len(text) if length is None else length) + text + data


def read_safetensors(blob: bytes):
    (n,) = struct.unpack("<Q", blob[:8])
    return json.loads(blob[8:8 + n]), blob[8 + n:]


# ---------------------------------------------------------------- seeds from the real vectors and from neyra

def vector_cases(kind: str) -> list[dict]:
    return [c for c in json.loads((VECTORS / "cases.json").read_text()) if c["kind"] == kind]


def read_fixture(rel: str) -> bytes:
    return (VECTORS / rel).read_bytes()


SECRET = bytes(range(32))
WRONG_SECRET = bytes(range(1, 33))


# ---------------------------------------------------------------- jcs, digest, tensor, rowtree

def gen_jcs(rng: random.Random, b: Bench, n: int) -> None:
    seeds = [bytes.fromhex(c["input_hex"]) for c in vector_cases("jcs")]
    for _ in range(n):
        mode = rng.random()
        if mode < 0.35:
            data, label = flip_bytes(rng, rng.choice(seeds))
        elif mode < 0.65:
            data, label = grammar_json(rng)
        elif mode < 0.8:
            depth = rng.choice([126, 127, 128, 129, 130, 200])
            shape = rng.choice(["[]", "{}", "mixed"])
            opens, closes = [], []
            for i in range(depth):
                o = "[" if shape == "[]" else '{"a":' if shape == "{}" else rng.choice(["[", '{"a":'])
                opens.append(o)
                closes.append("]" if o == "[" else "}")
            data, label = ("".join(opens) + "1" + "".join(reversed(closes))).encode(), f"nest:{depth}:{shape}"
            if rng.random() < 0.3:
                data = b" " * rng.randrange(3) + data + rng.choice([b"", b" ", b"\n", b"x", b" 1"])
        else:
            data, label = grammar_json(rng, scalar=True)
        b.add("jcs", label, input_hex=data.hex())


def grammar_json(rng: random.Random, scalar: bool = False) -> tuple[bytes, str]:
    def value(depth: int) -> str:
        r = rng.random()
        if depth > 3 or r < 0.35:
            return rng.choice(NUMBER_TOKENS if rng.random() < 0.5 else STRING_TOKENS)
        if r < 0.45:
            return rng.choice(["true", "false", "null", "True", "nul", "tru", "NULL"])
        sp = lambda: rng.choice(["", "", " ", "\n", rng.choice(WHITESPACE)])
        if r < 0.7:
            return "[" + sp() + ("," + sp()).join(value(depth + 1) for _ in range(rng.randrange(4))) + sp() + "]"
        keys = [rng.choice(KEY_POOL) for _ in range(rng.randrange(4))]
        body = ("," + sp()).join(f"{esc(k)}{sp()}:{sp()}{value(depth + 1)}" for k in keys)
        return "{" + sp() + body + sp() + "}"

    text = value(3 if scalar else 0)
    if rng.random() < 0.15:
        text = rng.choice(["\ufeff", " ", "\n", "/**/", "x", ""]) + text + rng.choice(["", " ", "\n", ",", " 1", "\x00", "}"])
    return text.encode("utf-8", "surrogatepass"), "grammar"


def gen_digest(rng: random.Random, b: Bench, n: int) -> None:
    sizes = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U16": 2, "I16": 2, "F16": 2,
             "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
    for _ in range(n):
        if rng.random() < 0.3:
            domain = rng.choice(["payload", "patch", "weights", "ledger", "holdout", "key", "rowleaf", "rownode",
                                 "rowtensor", "", "é", "a/b", "x\x00y"])
            b.add("digest", "digest", domain=domain, data_hex=rng.randbytes(rng.choice([0, 1, 31, 32, 55, 56, 63, 64, 65, 300])).hex())
        else:
            dtype = rng.choice(list(sizes))
            ndim = rng.choice([0, 1, 1, 2, 2, 3, 4, 255])
            shape = [rng.choice([0, 1, 1, 2, 3, 5]) for _ in range(ndim)]
            count = 1
            for d in shape:
                count *= d
            if count > 4096:
                shape, count = [3], 3
            b.add("tensor", f"tensor:ndim{ndim}", dtype=dtype, shape=shape, raw_hex=rng.randbytes(count * sizes[dtype]).hex())


def gen_rowtree(rng: random.Random, b: Bench, n: int) -> None:
    for _ in range(n):
        count = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255])
        b.add("rowtree", f"rowtree:{count}", leaves_hex=[rng.randbytes(32).hex() for _ in range(count)])


# ---------------------------------------------------------------- models for root and rowroot

DTYPES = {"U8": 1, "I8": 1, "BOOL": 1, "U16": 2, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U32": 4, "I64": 8, "F64": 8,
          "F8_E4M3": 1}
NAME_POOL = ["w", "model.layers.0.weight", "é.bias", "\ufb33", "\U0001f600", "\uffff", "\U00010000", "a", "A", "b",
             "", "x y", "a\nb", 'q"q', "back\\slash", "\x7f", "e\u0301", "\u2028"]


def model_files(rng: random.Random) -> tuple[list[tuple[str, bytes]], str]:
    """A well formed weights folder: one to three shards of up to four tensors each."""
    names = rng.sample(NAME_POOL, rng.randrange(1, 5))
    shards = [[] for _ in range(rng.choice([1, 1, 2, 3]))]
    for name in names:
        dtype = rng.choice(list(DTYPES))
        shape = [rng.choice([0, 1, 2, 3, 70]) for _ in range(rng.choice([0, 1, 1, 2, 3]))]
        count = 1
        for d in shape:
            count *= d
        shards[rng.randrange(len(shards))].append((name, dtype, shape, rng.randbytes(count * DTYPES[dtype])))
    files = []
    for i, tensors in enumerate(shards):
        header, data = {}, b""
        for name, dtype, shape, raw in tensors:
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
            data += raw
        if rng.random() < 0.3:
            header["__metadata__"] = {"format": "pt"}
        files.append((f"model-{i}.safetensors", safetensors(header, data)))
    return files, "model"


def damage_shard(rng: random.Random, blob: bytes) -> tuple[bytes, str]:
    """Break one safetensors file in a way a header parser might mishandle."""
    header, data = read_safetensors(blob)
    tensors = [k for k in header if k != "__metadata__"]
    ops = ["bytes", "length", "dtype", "shape", "offsets", "meta", "dupkey", "extra-key", "trailing", "pad", "bigshape",
           "negative", "bool-dim", "float-dim", "string-offset", "no-tensors", "meta-nonstring", "nest"]
    op = rng.choice(ops)
    if op == "bytes":
        return flip_bytes(rng, blob)[0], "st:bytes"
    if op == "length":
        n = struct.unpack("<Q", blob[:8])[0]
        new = rng.choice([n - 1, n + 1, n + 8, 0, 2**32, 2**63, 2**64 - 1, 100_000_000, 100_000_001, 100_000_000 + 8])
        return struct.pack("<Q", new % 2**64) + blob[8:], f"st:length={new}"
    if op == "trailing":
        return blob + rng.choice([b"\x00", b"x", b"\x00" * 8]), "st:trailing"
    if op == "no-tensors":
        return safetensors({}, b""), "st:empty-header"
    if op == "pad":
        n = struct.unpack("<Q", blob[:8])[0]
        pad = rng.choice([b"\x00", b"\t", b"\n", b" " * 8, b"\x00" * 8, b"\xff"])
        return safetensors({}, data, raw_header=blob[8:8 + n].rstrip(b" ") + pad), "st:pad"
    if op == "nest":
        return safetensors({}, b"", raw_header=b"[" * 200 + b"]" * 200), "st:nested-header"
    if op == "dupkey" and tensors and rng.random() < 0.5:
        n = struct.unpack("<Q", blob[:8])[0]
        raw = blob[8:8 + n].decode()
        k = rng.choice(tensors)
        field = rng.choice(['"dtype"', '"shape"', '"data_offsets"'])
        at_ = raw.index(field, raw.index(esc(k)))
        end = raw.index(",", at_) if field == '"dtype"' else raw.index("]", at_) + 1
        return safetensors({}, data, raw_header=(raw[:end] + "," + raw[at_:end] + raw[end:]).encode()), "st:nested-duplicate-key"
    if op == "dupkey" and tensors:
        n = struct.unpack("<Q", blob[:8])[0]
        raw = blob[8:8 + n].decode().rstrip()
        k = rng.choice(tensors)
        extra = json.dumps({k: header[k]}, ensure_ascii=False)[1:-1]
        return safetensors({}, data, raw_header=(raw[:-1] + "," + extra + "}").encode()), "st:duplicate-key"
    if op == "meta":
        header["__metadata__"] = rng.choice([[], "x", 1, None, {}, {"a": "b"}, {"a": {"b": "c"}}, {"a": 1}, {"a": None}])
        return safetensors(header, data), "st:metadata"
    if op == "meta-nonstring":
        header["__metadata__"] = {"format": rng.choice([1, True, None, ["x"]])}
        return safetensors(header, data), "st:metadata-type"
    if op == "extra-key":
        header[rng.choice(["__other__", "", "x"])] = rng.choice([{}, [], 1, None, {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]}])
        return safetensors(header, data), "st:extra-entry"
    if not tensors:
        return blob, "st:noop"
    k = rng.choice(tensors)
    e = header[k]
    if op == "dtype":
        e["dtype"] = rng.choice(["u8", "F32 ", "F4", "I128", "BF16", "U8", "", 1, None, "bool", "F8_E4M3FN", "C64"])
    elif op == "shape":
        e["shape"] = rng.choice([[], [1], [0], [2, 2], [1, 1, 1], "3", 3, None, [[1]]])
    elif op == "bigshape":
        e["shape"] = rng.choice([[2**32, 2**32], [2**63, 2], [2**64], [2**64 + 1], [2**31, 2**31, 2], [2**53], [2**32, 2**32 + 1]])
        e["data_offsets"] = rng.choice([[0, 0], e["data_offsets"], [0, 2**64 % 2**64]])
    elif op == "negative":
        if rng.random() < 0.5:
            e["shape"] = [-1]
        else:
            e["data_offsets"] = [-1, e["data_offsets"][1]]
    elif op == "bool-dim":
        e["shape"] = [True]
    elif op == "float-dim":
        e["shape"] = [1.0]
    elif op == "string-offset":
        e["data_offsets"] = [0, str(e["data_offsets"][1])]
    elif op == "offsets":
        a, z = e["data_offsets"]
        e["data_offsets"] = rng.choice([[a, z + 1], [a + 1, z], [z, a], [a], [a, z, z], [a, z + 2**64], [a, 2**64 - 1], [1.0, z]])
    return safetensors(header, data), f"st:{op}"


def gen_models(rng: random.Random, b: Bench, n: int) -> None:
    for _ in range(n):
        files, label = model_files(rng)
        d = b.folder()
        extra = rng.random()
        for name, blob in files:
            if rng.random() < 0.35:
                blob, label = damage_shard(rng, blob)
            (d / name).write_bytes(blob)
        if extra < 0.1:
            (d / "Case.SAFETENSORS").write_bytes(files[0][1])
            label += "+upper-ext"
        elif extra < 0.2:
            (d / ".hidden.safetensors").write_bytes(files[0][1])
            label += "+hidden"
        elif extra < 0.3:
            (d / "sub").mkdir()
            (d / "sub" / "x.safetensors").write_bytes(files[0][1])
            label += "+nested"
        elif extra < 0.4:
            (d / "notes.safetensors.bak").write_bytes(b"junk")
            (d / "config.json").write_text("{}")
            label += "+junk-files"
        elif extra < 0.45:
            (d / "dir.safetensors").mkdir()
            label += "+dir-named-safetensors"
        elif extra < 0.5:
            (d / "consolidated.safetensors").write_bytes(files[0][1])
            label += "+consolidated"
        elif extra < 0.55:
            (d / "extra.safetensors").write_bytes(safetensors({"z.extra": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, b"x"))
            label += "+extra-shard"
        rel = d.name
        kind = "root" if rng.random() < 0.5 else "rowroot"
        b.add(kind, f"{kind}:{label}", model=rel)


# ---------------------------------------------------------------- patches

def neyra_patch_seed(rng: random.Random, work: Path) -> tuple[Path, Path]:
    """A valid patch and its base folder, made with neyra's own producer."""
    import numpy as np

    from neyra.patch import Patch, diff_tensor, write_safetensors
    from neyra.protocol import tensor_table
    from neyra.store import Checkpoint

    base_dir = work / "base"
    base_dir.mkdir(parents=True)
    tensors, bits_as = {}, "BF16"
    names = rng.sample(NAME_POOL, rng.randrange(1, 5))
    for name in names:
        shape = tuple(rng.choice([1, 2, 3, 8, 17]) for _ in range(rng.choice([1, 2])))
        tensors[name] = np.array([rng.randrange(65536) for _ in range(int(np.prod(shape)))], dtype=np.uint16).reshape(shape)
    write_safetensors(base_dir / "model.safetensors", tensors, {"format": "pt"}, bits_as)
    ck = Checkpoint.open(str(base_dir))
    edits = []
    for name in rng.sample(names, rng.randrange(0, len(names) + 1)):
        edited = tensors[name].copy().reshape(-1)
        for i in rng.sample(range(edited.size), rng.randrange(0, min(edited.size, 6) + 1)):
            edited[i] ^= np.uint16(rng.randrange(1, 65536))
        edits.append(diff_tensor(name, bits_as, tensors[name], edited.reshape(tensors[name].shape)))
    patch = Patch(edits, {"note": "seed"}).bind(tensor_table(ck))
    out = work / "seed.neyra"
    patch.save(out)
    return out, base_dir


def reseal(manifest: dict, payloads: dict[str, bytes], loose: Loose | None = None, header_pad: bytes = b"",
           extra_meta: dict | None = None, text: str | None = None) -> bytes:
    """A patch file from a manifest and payload tensors. Payload digests are recomputed unless the caller broke them."""
    header, data = {"__metadata__": {"neyra.manifest": text if text is not None else (loose.dumps(manifest) if loose else jcs(manifest)),
                                     **(extra_meta or {})}}, b""
    for name, raw in payloads.items():
        dtype = "I64" if name.endswith(".idx") else {1: "U8", 2: "U16", 4: "U32", 8: "U64"}[width_of(manifest, name)]
        if name.endswith((".left", ".right")):
            dtype = "F64"
        header[name] = {"dtype": dtype, "shape": [len(raw) // {"I64": 8, "F64": 8}.get(dtype, int(dtype[1:]) // 8)],
                        "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    return safetensors(header, data, pad=header_pad)


def width_of(manifest: dict, payload: str) -> int:
    try:
        entry = manifest["tensors"][int(payload.split(".")[0])]
        return {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U16": 2, "I16": 2, "F16": 2,
                "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}[entry["dtype"]]
    except (KeyError, IndexError, ValueError, TypeError):
        return 2


def split_patch(blob: bytes) -> tuple[dict, dict[str, bytes]]:
    header, data = read_safetensors(blob)
    manifest = json.loads(header["__metadata__"]["neyra.manifest"])
    payloads = {k: data[v["data_offsets"][0]:v["data_offsets"][1]] for k, v in header.items() if k != "__metadata__"}
    return manifest, payloads


def fix_digests(manifest: dict, payloads: dict[str, bytes]) -> dict:
    manifest = clone(manifest)
    for i, t in enumerate(manifest["tensors"]):
        if isinstance(t, dict):
            for part, key in (("idx", "idx_digest"), ("val", "val_digest")):
                if f"{i}.{part}" in payloads:
                    t[key] = domain_digest("payload", payloads[f"{i}.{part}"])
    return manifest


def gen_patches(rng: random.Random, b: Bench, work: Path, n: int) -> None:
    seeds = []
    for s in range(6):
        folder = work / f"seed{s}"
        folder.mkdir()
        seeds.append(neyra_patch_seed(rng, folder))
    fixtures = [VECTORS / "fixtures" / "good.neyra"]
    base_for = {}
    for path, base in seeds:
        base_for[path] = base
    base_for[fixtures[0]] = VECTORS / "fixtures" / "base"
    pool = [*seeds, (fixtures[0], base_for[fixtures[0]])]
    for _ in range(n):
        path, base = rng.choice(pool)
        blob = path.read_bytes()
        manifest, payloads = split_patch(blob)
        with_base = rng.random() < 0.6
        mode = rng.choice(["bytes", "bytes", "header", "manifest", "manifest", "manifest-resealed", "payload",
                           "payload-resealed", "text", "meta", "order", "swapped-roots", "names"])
        label = mode
        if mode == "bytes":
            out, label = flip_bytes(rng, blob)
        elif mode == "header":
            out, label = damage_shard(rng, blob)
        elif mode in ("manifest", "manifest-resealed"):
            doc, what = mutate_json(rng, manifest)
            if mode == "manifest-resealed" and isinstance(doc, dict) and isinstance(doc.get("tensors"), list):
                doc = fix_digests(doc, payloads)
            out, label = reseal(doc, payloads), f"{mode}:{what}"
        elif mode in ("payload", "payload-resealed"):
            changed = dict(payloads)
            key = rng.choice(list(changed)) if changed else None
            if key:
                changed[key], what = flip_bytes(rng, changed[key])
                if rng.random() < 0.3 and changed[key]:
                    changed[key] = changed[key][:len(changed[key]) // 8 * 8]
                label = f"{mode}:{key}:{what}"
            doc = fix_digests(manifest, changed) if mode == "payload-resealed" else manifest
            out = reseal(doc, changed)
        elif mode == "text":
            loose = Loose(rng, shuffle=0.4, space=0.4, dup=0.3)
            out, label = reseal(manifest, payloads, loose), "text:non-canonical"
            if rng.random() < 0.5:
                out, label = reseal(manifest, payloads, text=jcs(manifest).replace(":", ": ", 1)), "text:one-space"
        elif mode == "meta":
            extra = {"neyra.meta": rng.choice(["", "{}", "not json", "\x00"]), rng.choice(["a", "neyra.x", ""]): "v"}
            out, label = reseal(manifest, payloads, extra_meta=extra), "meta:extra"
            if rng.random() < 0.3:
                out = reseal(manifest, payloads, extra_meta={"neyra.manifest": jcs(manifest)}), "meta:dup"
                out, label = out[0], out[1]
        elif mode == "order":
            ts = manifest.get("tensors", [])
            if len(ts) > 1:
                ts[0], ts[-1] = ts[-1], ts[0]
            out, label = reseal(manifest, payloads), "order:swapped-entries"
        elif mode == "swapped-roots":
            manifest["base_root"], manifest["result_root"] = manifest["result_root"], manifest["base_root"]
            out, label = reseal(manifest, payloads), "swapped-roots"
        else:
            if manifest.get("tensors"):
                manifest["tensors"][0]["name"] = rng.choice(NAME_POOL)
            out, label = reseal(manifest, payloads), "names:rename-first"
        fields = {"file": b.put(out, ".neyra")}
        if with_base:
            fields["base"] = b.share(Path(base))
        b.add("patch", f"patch:{label}", **fields)


# ---------------------------------------------------------------- ledgers

def seed_ledger(rng: random.Random, length: int) -> list[dict]:
    entries, head = [], "sha256:" + "0" * 64
    for i in range(length):
        e = {"seq": i, "prev": head, "time": f"2026-11-04T12:0{i % 10}:00Z", "kind": rng.choice(["open", "sample", "é"]),
             "body": rng.choice([{}, {"n": i}, {"metric": "0.5"}, {"k": [1, "x", None, True]}])}
        entries.append(e)
        head = domain_digest("ledger", jcs(e).encode())
    return entries


def chain(entries: list[dict], from_index: int = 0) -> list[dict]:
    """Recompute prev from from_index on, so an edit to one entry does not also break every later link."""
    entries = clone(entries)
    for i in range(max(from_index, 1), len(entries)):
        if isinstance(entries[i - 1], dict) and isinstance(entries[i], dict):
            entries[i]["prev"] = domain_digest("ledger", jcs(entries[i - 1]).encode())
    return entries


def ledger_bytes(entries: list[dict]) -> bytes:
    return b"".join(jcs(e).encode("utf-8", "surrogatepass") + b"\n" for e in entries)


TIMES = ["2026-11-04T12:00:00Z", "2026-13-45T25:61:61Z", "0000-00-00T00:00:00Z", "2026-11-04t12:00:00Z",
         "2026-11-04T12:00:00z", "2026-11-04T12:00:00.5Z", "2026-11-04T12:00:00+00:00", "2026-11-04T12:00:00Z ",
         " 2026-11-04T12:00:00Z", "2026-11-04T12:00:00Z\n", "2026-11-04T12:00:00Z\x00", "２０２６-11-04T12:00:00Z",
         "2026-11-04T12:00:00Z\r\n", "2026-11-04 12:00:00Z", "٢٠٢٦-11-04T12:00:00Z", "2026-11-04T12:00:00", "", "20261104T120000Z",
         "10000-11-04T12:00:00Z", "2026-1-04T12:00:00Z", "2026-02-30T12:00:00Z", "9999-99-99T99:99:99Z"]
SEQ_VALUES = [True, False, -1, 1.0, "0", "1", None, 2**53 - 1, 2**53, [0], {}, 0, 1, 2]


def gen_ledgers(rng: random.Random, b: Bench, n: int) -> None:
    for _ in range(n):
        entries = seed_ledger(rng, rng.choice([0, 1, 2, 3, 5]))
        mode = rng.choice(["field", "field", "time", "seq", "lines", "bytes", "text", "resealed-field", "prev", "members"])
        label, raw = mode, None
        if mode == "time" and entries:
            i = rng.randrange(len(entries))
            entries[i]["time"] = rng.choice(TIMES)
            label = f"time:{entries[i]['time']!r}"
            entries = chain(entries, i + 1)
        elif mode == "seq" and entries:
            i = rng.randrange(len(entries))
            entries[i]["seq"] = rng.choice(SEQ_VALUES)
            label = f"seq:{entries[i]['seq']!r}@{i}"
            entries = chain(entries, i + 1)
        elif mode == "prev" and entries:
            i = rng.randrange(len(entries))
            entries[i]["prev"] = rng.choice(["sha256:" + "0" * 64, "sha256:" + "A" * 64, "sha256:" + "0" * 63, "", None, 1,
                                             domain_digest("ledger", b"x"), "sha256:" + "0" * 64 + "\n"])
            label = f"prev:{entries[i]['prev']!r}@{i}"
            entries = chain(entries, i + 1)
        elif mode == "members" and entries:
            i = rng.randrange(len(entries))
            op = rng.choice(["drop", "add", "retype"])
            if op == "drop":
                entries[i].pop(rng.choice(["seq", "prev", "time", "kind", "body"]))
            elif op == "add":
                entries[i][rng.choice(["x", "x-note", "extra", "Seq"])] = rng.choice([1, "a", None])
            else:
                entries[i][rng.choice(["kind", "body"])] = rng.choice([1, None, [], "s", {}, True])
            label = f"members:{op}"
            entries = chain(entries, i + 1)
        elif mode in ("field", "resealed-field") and entries:
            doc, what = mutate_json(rng, entries)
            entries = chain(doc, 0) if mode == "resealed-field" and isinstance(doc, list) else doc
            label = f"{mode}:{what}"
            if not isinstance(entries, list):
                entries = [entries]
        if mode == "lines":
            lines = ledger_bytes(entries).split(b"\n")[:-1]
            op = rng.choice(["blank", "dup", "swap", "drop", "crlf", "no-final", "extra-final", "garbage-tail", "space-prefix", "bom"])
            if op == "blank":
                lines.insert(rng.randrange(len(lines) + 1), b"")
            elif op == "dup" and lines:
                lines.insert(rng.randrange(len(lines)), rng.choice(lines))
            elif op == "swap" and len(lines) > 1:
                lines[0], lines[-1] = lines[-1], lines[0]
            elif op == "drop" and lines:
                lines.pop(rng.randrange(len(lines)))
            elif op == "space-prefix" and lines:
                lines[0] = b" " + lines[0]
            elif op == "bom" and lines:
                lines[0] = b"\xef\xbb\xbf" + lines[0]
            body = b"\n".join(lines) + (b"\n" if lines else b"")
            raw = {"crlf": body.replace(b"\n", b"\r\n"), "no-final": body.rstrip(b"\n"), "extra-final": body + b"\n",
                   "garbage-tail": body + b"{", }.get(op, body)
            label = f"lines:{op}"
        elif mode == "bytes":
            raw, label = flip_bytes(rng, ledger_bytes(entries))
        elif mode == "text":
            loose = Loose(rng, shuffle=0.5, space=0.5, dup=0.3)
            raw = b"".join(loose.dumps(e).encode() + b"\n" for e in entries)
            label = "text:non-canonical"
        b.add("ledger", f"ledger:{label}", file=b.put(raw if raw is not None else ledger_bytes(entries), ".jsonl"))


# ---------------------------------------------------------------- receipts and bindings

def claim_for(patch_id: str, result_root: str, rng: random.Random) -> dict:
    return {"type": "neyra.claim/1", "base_root": domain_digest("weights", b"b"), "result_root": result_root,
            "patch_id": patch_id, "task": {"id": "t"}, "eval_env": {"runtime": "fuzz"}, "gates": {"effect": True},
            "limits": {}, "statements": [], "verdict": rng.choice(["PASS", "FAIL"])}


def statement_of(claim: dict, head: str, length) -> dict:
    return {"_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": "result-weights", "digest": {"sha256": str(claim["result_root"]).split(":", 1)[-1]}},
                        {"name": "patch", "digest": {"sha256": str(claim["patch_id"]).split(":", 1)[-1]}}],
            "predicateType": "https://neyra.run/receipt/v1",
            "predicate": {"claim": claim, "digest_algorithm": "neyra/weights/v1", "ledger_head": head, "ledger_len": length}}


PAYLOAD_TYPE = "application/vnd.in-toto+json"


def envelope_text(statement, secret=SECRET, rng=None, canonical=True, payload_type=PAYLOAD_TYPE) -> dict:
    payload = (jcs(statement) if canonical else Loose(rng, shuffle=0.5, space=0.5).dumps(statement)).encode()
    keyid, sig = sign(secret, payload_type, payload)
    return {"payloadType": payload_type, "payload": b64(payload), "signatures": [{"keyid": keyid, "sig": sig}]}


def b64_variants(rng: random.Random, raw: bytes) -> tuple[str, str]:
    text = b64(raw)
    op = rng.choice(["unpadded", "urlsafe", "newline", "space", "double-pad", "trailing-bits", "prefix-space", "empty",
                     "nonascii", "tab", "crlf-wrap"])
    if op == "unpadded":
        return text.rstrip("="), op
    if op == "urlsafe":
        return text.replace("+", "-").replace("/", "_"), op
    if op == "newline":
        return text[:20] + "\n" + text[20:], op
    if op == "space":
        return text[:10] + " " + text[10:], op
    if op == "double-pad":
        return text + "=", op
    if op == "prefix-space":
        return " " + text, op
    if op == "empty":
        return "", op
    if op == "nonascii":
        return text[:-4] + "ＡＡＡＡ", op
    if op == "tab":
        return text + "\t", op
    if op == "crlf-wrap":
        return "\r\n".join(text[i:i + 76] for i in range(0, len(text), 76)), op
    # Change only bits that decode to nothing: the last character of a padded group.
    if text.endswith("=="):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        i = alphabet.index(text[-3])
        return text[:-3] + alphabet[i ^ 1] + "==", "trailing-bits"
    if text.endswith("="):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        i = alphabet.index(text[-2])
        return text[:-2] + alphabet[i ^ 1] + "=", "trailing-bits"
    return text, "trailing-bits-none"


class ReceiptWorld:
    """One consistent patch, ledger and receipt to mutate pieces of."""

    def __init__(self, rng: random.Random):
        self.good = VECTORS / "fixtures"
        patch_blob = (self.good / "good.neyra").read_bytes()
        manifest, _ = split_patch(patch_blob)
        self.patch_id = domain_digest("patch", jcs(manifest).encode())
        self.result_root = manifest["result_root"]
        self.entries = [json.loads(line) for line in (self.good / "ledger-good.jsonl").read_text().splitlines()]
        self.head = domain_digest("ledger", jcs(self.entries[-1]).encode())
        self.claim = claim_for(self.patch_id, self.result_root, rng)
        self.statement = statement_of(self.claim, self.head, len(self.entries))
        self.public = public_hex(SECRET)


def gen_receipts(rng: random.Random, b: Bench, n: int) -> None:
    w = ReceiptWorld(rng)
    for _ in range(n):
        mode = rng.choice(["statement", "statement", "statement-text", "envelope", "payload-b64", "sig-b64", "signatures",
                           "bytes", "key", "claim-member", "subject"])
        statement, public, label = clone(w.statement), w.public, mode
        env = None
        if mode == "statement":
            statement, what = mutate_json(rng, statement)
            label = f"statement:{what}"
        elif mode == "statement-text":
            env, label = envelope_text(statement, rng=rng, canonical=False), "statement:non-canonical"
        elif mode == "claim-member":
            claim = statement["predicate"]["claim"]
            op = rng.choice(["drop", "retype", "extra"])
            key = rng.choice(list(claim))
            if op == "drop":
                del claim[key]
            elif op == "retype":
                claim[key] = rng.choice(ATOMS)
            else:
                claim[rng.choice(["x-note", "extra", "A"])] = rng.choice(ATOMS)
            label = f"claim:{op}:{key}"
        elif mode == "subject":
            op = rng.choice(["reverse", "dup", "uppercase", "short", "extra", "name", "prefix"])
            sub = statement["subject"]
            if op == "reverse":
                sub.reverse()
            elif op == "dup":
                sub[1] = clone(sub[0])
            elif op == "uppercase":
                sub[0]["digest"]["sha256"] = sub[0]["digest"]["sha256"].upper()
            elif op == "short":
                sub[0]["digest"]["sha256"] = sub[0]["digest"]["sha256"][:-1]
            elif op == "extra":
                sub[0]["digest"]["sha512"] = "00"
            elif op == "name":
                sub[0]["name"] = "Result-weights"
            else:
                sub[0]["digest"]["sha256"] = "sha256:" + sub[0]["digest"]["sha256"]
            label = f"subject:{op}"
        if env is None:
            env = envelope_text(statement)
        if mode == "envelope":
            env, what = mutate_json(rng, env)
            label = f"envelope:{what}"
        elif mode == "payload-b64":
            env["payload"], what = b64_variants(rng, jcs(statement).encode())
            label = f"payload-b64:{what}"
        elif mode == "sig-b64":
            env["signatures"][0]["sig"], what = b64_variants(rng, base64.b64decode(env["signatures"][0]["sig"]))
            label = f"sig-b64:{what}"
        elif mode == "signatures":
            good = env["signatures"][0]
            keyid_other, sig_other = sign(WRONG_SECRET, PAYLOAD_TYPE, base64.b64decode(env["payload"]))
            bad = {"keyid": good["keyid"], "sig": b64(bytes(64))}
            options = {
                "bad-then-good": [bad, good], "good-then-bad": [good, bad], "other-then-good": [{"keyid": keyid_other, "sig": sig_other}, good],
                "none": [], "not-a-list": {"keyid": good["keyid"], "sig": good["sig"]}, "non-object": [1, good],
                "null-entry": [None, good], "missing-keyid": [{"sig": good["sig"]}], "missing-sig": [{"keyid": good["keyid"]}],
                "keyid-upper": [{**good, "keyid": good["keyid"].upper()}], "extra-member": [{**good, "x": 1}],
                "short-sig": [{**good, "sig": b64(base64.b64decode(good["sig"])[:63])}],
                "long-sig": [{**good, "sig": b64(base64.b64decode(good["sig"]) + b"\x00")}],
                "sig-not-string": [{**good, "sig": 1}], "keyid-not-string": [{"keyid": 1, "sig": good["sig"]}],
                "good-wrong-keyid": [{**good, "keyid": keyid_other}],
            }
            op = rng.choice(list(options))
            env["signatures"] = options[op]
            label = f"signatures:{op}"
        elif mode == "key":
            op = rng.choice(["wrong", "none", "short", "long", "zero"])
            public = {"wrong": public_hex(WRONG_SECRET), "none": None, "short": w.public[:-2], "long": w.public + "00",
                      "zero": "00" * 32}[op]
            label = f"key:{op}"
        if mode == "envelope" and isinstance(env, dict) and rng.random() < 0.2:
            env["payloadType"] = rng.choice(["Application/vnd.in-toto+json", "application/vnd.in-toto+json ", "text/plain", ""])
        text = json.dumps(env) if rng.random() < 0.8 else Loose(rng, space=0.5, dup=0.2).dumps(env)
        raw = text.encode()
        if mode == "bytes":
            raw, label = flip_bytes(rng, raw)
        if public is not None and len(public) != 64:
            continue
        fields = {"file": b.put(raw, ".json")}
        if public:
            fields["public_hex"] = public
        b.add("receipt", f"receipt:{label}", **fields)


def gen_bindings(rng: random.Random, b: Bench, n: int) -> None:
    w = ReceiptWorld(rng)
    good_patch = w.good / "good.neyra"
    other_patch = w.good / "other-patch.neyra"
    for _ in range(n):
        entries = clone(w.entries)
        head, length = w.head, len(entries)
        mode = rng.choice(["head", "length", "types", "ledger", "patch", "claim"])
        label = mode
        statement_head, statement_len = head, length
        claim = clone(w.claim)
        if mode in ("length", "head"):
            i = rng.randrange(len(entries))
            statement_head = domain_digest("ledger", jcs(entries[i]).encode())
            statement_len = rng.choice([i + 1, i + 1, i, i + 2, 0, -1, True, False, "3", None, [3], 2**53 - 1])
            if mode == "head":
                statement_head = rng.choice([statement_head, "sha256:" + "0" * 64, None, 1, "", head.upper(), head + " ",
                                             domain_digest("ledger", b"?")])
        elif mode == "types":
            statement_head = rng.choice([None, 3, True, [head], {"a": head}])
            statement_len = rng.choice([None, "3", True, [3], 3])
        elif mode == "claim":
            claim["patch_id"] = rng.choice([w.patch_id.upper(), domain_digest("patch", b"x"), w.patch_id + "x", None, 123, w.patch_id[7:]])
            claim["result_root"] = rng.choice([claim["result_root"], domain_digest("weights", b"x"), 7, claim["result_root"][7:]])
        label = f"{mode}:{statement_head!r:.20}:{statement_len!r}"
        statement = statement_of(claim, statement_head, statement_len)
        if mode == "claim":
            # The subjects follow the claim's text, even when it is not a digest, to test how strictly they are compared.
            statement["subject"][1]["digest"]["sha256"] = str(claim["patch_id"]).split(":", 1)[-1]
            statement["subject"][0]["digest"]["sha256"] = str(claim["result_root"]).split(":", 1)[-1]
        if mode == "ledger":
            op = rng.choice(["tail", "head-only", "reorder", "empty", "extra-entry"])
            if op == "tail":
                entries = entries[:2]
            elif op == "head-only":
                entries = entries[:1]
            elif op == "reorder":
                entries.reverse()
            elif op == "empty":
                entries = []
            else:
                entries.append({**entries[-1], "seq": 3, "prev": w.head})
            label = f"ledger:{op}"
        env = envelope_text(statement_of(claim, statement_head, statement_len) if mode != "claim" else statement)
        patch = good_patch if mode != "patch" or rng.random() < 0.3 else other_patch
        if mode == "patch":
            label = "patch:other" if patch == other_patch else "patch:same"
        b.add("binding", f"binding:{label}", receipt=b.put(json.dumps(env).encode(), ".json"),
              ledger=b.put(ledger_bytes(entries), ".jsonl"), patch=b.share(patch))


# ---------------------------------------------------------------- row proofs

def gen_rowproofs(rng: random.Random, b: Bench, work: Path, n: int) -> None:
    import numpy as np

    from neyra.patch import write_safetensors
    from neyra.rows import build_index, fetch, prove
    from neyra.store import Checkpoint

    model = work / "rows-model"
    model.mkdir()
    tensors = {
        "grid": np.array([rng.randrange(65536) for _ in range(70 * 33)], dtype=np.uint16).reshape(70, 33),
        "tall": np.array([rng.randrange(65536) for _ in range(3000 * 2)], dtype=np.uint16).reshape(3000, 2),
        "vec": np.array([rng.randrange(65536) for _ in range(5000)], dtype=np.uint16),
        "one": np.array(7, dtype=np.uint16), "tiny": np.array([[1, 2, 3]], dtype=np.uint16),
    }
    write_safetensors(model / "model.safetensors", tensors, {"format": "pt"}, "BF16")
    ck = Checkpoint.open(str(model))
    index = build_index(ck)
    seeds = []
    for name, arr in tensors.items():
        rows = arr.shape[0] if arr.ndim else 1
        for _ in range(4):
            a = rng.randrange(rows)
            z = rng.randrange(a + 1, min(rows, a + 40) + 1)
            ranges = [(a, z)]
            if rows > 100 and rng.random() < 0.5:
                c = rng.randrange(z, rows)
                ranges.append((c, min(rows, c + rng.randrange(1, 10))))
            proof = prove(index, name, ranges)
            seeds.append((proof, fetch(ck.tensors[name], proof)))
    for _ in range(n):
        proof, data = rng.choice(seeds)
        proof = clone(proof)
        mode = rng.choice(["proof", "proof", "data", "text", "lists", "ranges", "ints", "bytes"])
        label = mode
        if mode == "proof":
            proof, what = mutate_json(rng, proof)
            label = f"proof:{what}"
        elif mode == "data":
            data, what = flip_bytes(rng, data)
            label = f"data:{what}"
        elif mode == "lists":
            key = rng.choice(["tree", "path"])
            hashes = proof.get(key, [])
            op = rng.choice(["swap", "dup", "drop", "append", "zero", "upper"])
            if op == "swap" and len(hashes) > 1:
                hashes[0], hashes[-1] = hashes[-1], hashes[0]
            elif op == "dup" and hashes:
                hashes.append(hashes[0])
            elif op == "drop" and hashes:
                hashes.pop()
            elif op == "append":
                hashes.append("sha256:" + "0" * 64)
            elif op == "zero" and hashes:
                hashes[0] = "sha256:" + "0" * 64
            elif op == "upper" and hashes:
                hashes[0] = hashes[0].upper().replace("SHA256", "sha256")
            label = f"lists:{key}:{op}"
        elif mode == "ranges":
            ranges = proof["ranges"]
            op = rng.choice(["empty-list", "empty-range", "reverse", "negative", "float", "bool", "touching", "overlap",
                             "huge", "triple", "string", "past-end", "unsorted"])
            a, z = ranges[0]
            proof["ranges"] = {
                "empty-list": [], "empty-range": [[a, a]], "reverse": [[z, a]], "negative": [[-1, z]], "float": [[float(a), z]],
                "bool": [[False, True]], "touching": [[a, a + 1], [a + 1, z + 1]], "overlap": [[a, z], [a, z]],
                "huge": [[a, 2**64]], "triple": [[a, z, z]], "string": [[str(a), z]], "past-end": [[a, z + 10_000]],
                "unsorted": [[z + 5, z + 6], [a, z]],
            }[op]
            label = f"ranges:{op}"
        elif mode == "ints":
            key = rng.choice(["index", "count"])
            proof[key] = rng.choice([-1, 0, 1, 2, 5, 6, 2**32, 2**64, True, 1, "1", None])
            label = f"ints:{key}={proof[key]!r}"
        text = json.dumps(proof) if mode != "text" else Loose(rng, shuffle=0.5, space=0.5, dup=0.3).dumps(proof)
        raw = text.encode()
        if mode == "bytes":
            raw, label = flip_bytes(rng, raw)
        b.add("rowproof", f"rowproof:{label}", proof=b.put(raw, ".json"), data=b.put(data, ".data"))


# ---------------------------------------------------------------- running both verifiers

@dataclass
class Result:
    code: str
    value: str


def python_results(root: Path, cases: list[dict]) -> list[Result]:
    from neyra.conformance import _run

    out = []
    for case in cases:
        try:
            r = _run(root, case)
            out.append(Result(r.get("code", ""), r.get("hex") or r.get("digest") or r.get("root") or r.get("head") or ""))
        except Exception as e:  # noqa: BLE001 - a crash is the finding
            out.append(Result(f"crash:{type(e).__name__}", str(e)[:120]))
    return out


def go_results(binary: Path, root: Path) -> list[Result]:
    done = subprocess.run([str(binary), "run", str(root)], capture_output=True, text=True, check=False, timeout=900)
    if done.returncode != 0:
        raise SystemExit(f"go verifier failed: {done.stderr[:500]}")
    return [Result(r["code"], r["value"]) for r in json.loads(done.stdout)]


def classify(case: dict, py: Result, go: Result, raw_hint: bytes) -> str | None:
    """The kind of disagreement, or None when the two agree or the spec allows the difference."""
    if py == go:
        return None
    if go.code == "unknown_extension" and b"low-rank/1" in raw_hint:
        return "declined"
    py_ok = py.code == "ok" or (py.code == "" and py.value != "")
    go_ok = go.code == "ok" or (go.code == "" and go.value != "")
    if py.code.startswith("crash") or go.code.startswith("panic"):
        return "crash"
    if py_ok != go_ok:
        return "accepted_by_one"
    if py_ok and go_ok:
        return "value_differs"
    return "code_differs"


def case_files(root: Path, case: dict) -> list[str]:
    return [case[k] for k in ("file", "proof", "data", "receipt", "ledger", "patch", "model", "base") if case.get(k)]


def hint_bytes(root: Path, case: dict) -> bytes:
    data = b""
    for key in ("file", "patch"):
        p = case.get(key)
        if p and (root / p).is_file():
            data += (root / p).read_bytes()
        elif p and Path(p).is_file():
            data += Path(p).read_bytes()
    return data


def keep(root: Path, out: Path, case: dict, py: Result, go: Result, kind: str) -> None:
    folder = out / kind / f"{case['kind']}-{case['id']}-{abs(hash(case['label'])) % 10**6}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "case.json").write_text(json.dumps({"case": case, "python": vars(py), "go": vars(go)}, indent=1))
    for rel in case_files(root, case):
        src = root / rel
        if src.is_file():
            shutil.copy(src, folder / Path(rel).name)
        elif src.is_dir():
            shutil.copytree(src, folder / Path(rel).name, dirs_exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=5000, help="mutants per batch")
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="findings", help="where disagreements are saved")
    ap.add_argument("--only", help="comma separated kinds: jcs,digest,rowtree,models,patch,ledger,receipt,binding,rowproof")
    ap.add_argument("--examples", type=int, default=3, help="examples kept per distinct disagreement")
    a = ap.parse_args(argv)

    out = Path(a.out)
    wanted = set(a.only.split(",")) if a.only else None
    scratch = Path(tempfile.mkdtemp(prefix="neyra-fuzz-"))
    try:
        binary = scratch / "neyra-verify"
        built = subprocess.run(["go", "build", "-o", str(binary), "."], cwd=GO_DIR, capture_output=True, text=True, check=False)
        if built.returncode:
            raise SystemExit(f"go build failed: {built.stderr}")
        seen: collections.Counter = collections.Counter()
        kept: collections.Counter = collections.Counter()
        first: dict = {}
        totals = collections.Counter()
        for batch in range(a.batches):
            rng = random.Random(a.seed * 1000 + batch)
            root = scratch / f"batch{batch}"
            bench = Bench(root)
            work = scratch / f"work{batch}"
            work.mkdir()
            share = {"jcs": 0.22, "digest": 0.05, "rowtree": 0.03, "models": 0.12, "patch": 0.2, "ledger": 0.1,
                     "receipt": 0.12, "binding": 0.06, "rowproof": 0.1}
            for name, fraction in share.items():
                if wanted and name not in wanted:
                    continue
                n = max(1, int(a.cases * fraction))
                if name == "jcs":
                    gen_jcs(rng, bench, n)
                elif name == "digest":
                    gen_digest(rng, bench, n)
                elif name == "rowtree":
                    gen_rowtree(rng, bench, n)
                elif name == "models":
                    gen_models(rng, bench, n)
                elif name == "patch":
                    gen_patches(rng, bench, work, n)
                elif name == "ledger":
                    gen_ledgers(rng, bench, n)
                elif name == "receipt":
                    gen_receipts(rng, bench, n)
                elif name == "binding":
                    gen_bindings(rng, bench, n)
                else:
                    gen_rowproofs(rng, bench, work, n)
            bench.save()
            py, go = python_results(root, bench.cases), go_results(binary, root)
            for case, p, g in zip(bench.cases, py, go):
                totals["cases"] += 1
                totals[f"kind:{case['kind']}"] += 1
                kind = classify(case, p, g, hint_bytes(root, case))
                if kind is None:
                    accepted = p.code == "ok" or (p.code == "" and p.value != "")
                    totals[f"agree-{'accept' if accepted else 'reject'}:{case['kind']}"] += 1
                    continue
                if kind == "declined":
                    totals["declined"] += 1
                    continue
                signature = (kind, case["kind"], case["label"].split(":", 1)[0] + ":" + (case["label"].split(":")[1] if ":" in case["label"] else ""),
                             p.code, g.code)
                seen[signature] += 1
                totals["disagreements"] += 1
                if kept[signature] < a.examples:
                    kept[signature] += 1
                    first.setdefault(signature, (case, p, g))
                    keep(root, out, case, p, g, kind)
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"{totals['cases']} mutants, {totals['disagreements']} disagreements in {len(seen)} distinct kinds, "
          f"{totals['declined']} declined (optional extension)")
    verdicts = sorted(k for k in totals if k.startswith("agree-"))
    print("agreed: " + ", ".join(f"{k[6:]}={totals[k]}" for k in verdicts))
    for rank in RANKS:
        rows = sorted(((s, c) for s, c in seen.items() if s[0] == rank), key=lambda x: -x[1])
        for (kind, case_kind, label, pc, gc), count in rows:
            case, p, g = first[(kind, case_kind, label, pc, gc)]
            print(f"{kind:16} {case_kind:9} {count:5}x  {case['label'][:70]:70}  python={(p.code or p.value)[:40]!r} go={(g.code or g.value)[:40]!r}")
    return 1 if seen else 0


if __name__ == "__main__":
    sys.exit(main())
