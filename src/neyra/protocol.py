"""Neyra Protocol v1: canonical encoding, digests, claims and receipts.

The normative text is spec/PROTOCOL.md. Nothing hashed here contains a float, because
two languages cannot be trusted to print one the same way.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .store import CHUNK, Checkpoint, TensorRef

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://neyra.run/receipt/v1"
PAYLOAD_TYPE = "application/vnd.in-toto+json"
MAX_SAFE = 2**53 - 1
MAX_DEPTH = 128
CLAIM_MEMBERS = {"type", "base_root", "result_root", "patch_id", "task", "eval_env", "statements", "gates",
                 "verdict", "limits"}


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


def _string(s: str) -> str:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ProtocolError("bad_json", "lone surrogate in a string") from e
    return json.dumps(s, ensure_ascii=False)


def _canonical(value) -> str:
    if value is None or value is True or value is False:
        return json.dumps(value)
    if isinstance(value, bool):
        return json.dumps(bool(value))
    if isinstance(value, int):
        if abs(value) > MAX_SAFE:
            raise ProtocolError("bad_json", f"{value} is outside the exact integer range")
        return str(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ProtocolError("bad_json", "object keys must be strings")
        keys = sorted(value, key=lambda k: k.encode("utf-16-be", "surrogatepass"))
        return "{" + ",".join(f"{_string(k)}:{_canonical(value[k])}" for k in keys) + "}"
    raise ProtocolError("bad_json", f"{type(value).__name__} cannot appear in a canonical object")


def jcs(value) -> bytes:
    """RFC 8785 for the subset this protocol allows: no floats anywhere."""
    return _canonical(value).encode("utf-8")


def strict_loads(raw: bytes | str):
    """Parse JSON, refusing duplicate keys, non finite numbers and extreme nesting."""

    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ProtocolError("bad_json", f"duplicate key {k}")
            out[k] = v
        return out

    def no_constant(name):
        raise ProtocolError("bad_json", f"{name} is not JSON")

    try:
        text = raw.decode("utf-8", "strict") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as e:
        raise ProtocolError("bad_json", "invalid UTF-8") from e
    depth = deepest = 0
    quoted = escaped = False
    for ch in text:
        if quoted:
            quoted = escaped or ch != '"'
            escaped = not escaped and ch == "\\"
        elif ch == '"':
            quoted = True
        elif ch in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif ch in "]}":
            depth -= 1
    if deepest > MAX_DEPTH:
        raise ProtocolError("bad_json", "nested too deeply")
    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=no_constant)
    except (ValueError, RecursionError) as e:
        if isinstance(e, ProtocolError):
            raise
        raise ProtocolError("bad_json", str(e)) from e


def digest(domain: str, data: bytes) -> str:
    return "sha256:" + hashlib.sha256(f"neyra/{domain}/v1\0".encode() + data).hexdigest()


def tensor_prefix(dtype: str, shape: tuple[int, ...]) -> bytes:
    head = b"neyra/tensor/v1\0" + dtype.encode() + b"\0" + bytes([len(shape)])
    return head + b"".join(struct.pack("<Q", d) for d in shape)


def tensor_digest(dtype: str, shape: tuple[int, ...], raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(tensor_prefix(dtype, shape) + raw).hexdigest()


def bits_digest(dtype: str, bits: np.ndarray) -> str:
    return tensor_digest(dtype, tuple(bits.shape), np.ascontiguousarray(bits).tobytes())


def ref_digest(ref: TensorRef) -> str:
    h = hashlib.sha256(tensor_prefix(ref.dtype, ref.shape))
    for o in range(ref.start, ref.end, CHUNK):
        h.update(ref.blob.read(o, min(CHUNK, ref.end - o)))
    return "sha256:" + h.hexdigest()


Table = dict[str, dict]


def tensor_table(ck: Checkpoint) -> Table:
    return {n: {"name": n, "dtype": r.dtype, "shape": list(r.shape), "digest": ref_digest(r)}
            for n, r in ck.tensors.items()}


def root_of(table: Table) -> str:
    rows = sorted(table.values(), key=lambda t: t["name"].encode("utf-16-be", "surrogatepass"))
    return digest("weights", jcs(rows))


def weights_root(ck: Checkpoint) -> str:
    return root_of(tensor_table(ck))


def decimal(x: float, places: int = 6) -> str:
    """A metric as a string. Floats never enter hashed content."""
    x = float(x)
    if not math.isfinite(x):
        raise ProtocolError("bad_json", "non finite metric")
    text = f"{x:.{places}f}"
    return "0." + "0" * places if float(text) == 0 else text


def plain(value):
    """Turn lab data into what canonical JSON allows."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return decimal(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, str):
        return value
    raise ProtocolError("bad_json", f"{type(value).__name__} has no canonical form")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def statement_for(claim: dict, patch_id: str, ledger_head: str, ledger_len: int) -> dict:
    return {
        "_type": STATEMENT_TYPE,
        "subject": [
            {"name": "result-weights", "digest": {"sha256": claim["result_root"].split(":", 1)[1]}},
            {"name": "patch", "digest": {"sha256": patch_id.split(":", 1)[1]}},
        ],
        "predicateType": PREDICATE_TYPE,
        "predicate": {"claim": claim, "digest_algorithm": "neyra/weights/v1", "ledger_head": ledger_head,
                      "ledger_len": ledger_len},
    }


def pae(payload_type: str, payload: bytes) -> bytes:
    t = payload_type.encode()
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def envelope(statement: dict, key: bytes | None = None) -> dict:
    """A DSSE envelope. Without a key it is an unauthenticated record, and says so by being empty."""
    payload = jcs(statement)
    signatures = []
    if key is not None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        private = Ed25519PrivateKey.from_private_bytes(key)
        public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        signatures.append({"keyid": digest("key", public),
                           "sig": base64.b64encode(private.sign(pae(PAYLOAD_TYPE, payload))).decode()})
    return {"payloadType": PAYLOAD_TYPE, "payload": base64.b64encode(payload).decode(), "signatures": signatures}


@dataclass
class Opened:
    statement: dict
    signed: bool


def strict_b64(text) -> bytes:
    """Standard alphabet, padding required, no whitespace, and no stray bits in the last character."""
    raw = base64.b64decode(text, validate=True)
    if base64.b64encode(raw).decode() != text:
        raise ValueError("not the canonical encoding")
    return raw


def decode_envelope(doc) -> tuple[dict, bytes]:
    signatures = doc.get("signatures") if isinstance(doc, dict) else None
    if (not isinstance(signatures, list) or not isinstance(doc.get("payloadType"), str)
            or not isinstance(doc.get("payload"), str)
            or not all(isinstance(s, dict) and isinstance(s.get("keyid"), str) and isinstance(s.get("sig"), str)
                       for s in signatures)):
        raise ProtocolError("bad_json", "not a DSSE envelope")
    if doc["payloadType"] != PAYLOAD_TYPE:
        raise ProtocolError("bad_type", "not an in-toto payload")
    try:
        payload = strict_b64(doc["payload"])
    except ValueError as e:
        raise ProtocolError("bad_json", "payload is not base64") from e
    statement = strict_loads(payload)
    if jcs(statement) != payload:
        raise ProtocolError("not_canonical", "payload is not in canonical form")
    return statement, payload


def check_signature(doc: dict, payload: bytes, public: bytes) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    wanted = digest("key", public)
    key = Ed25519PublicKey.from_public_bytes(public)
    for s in doc["signatures"]:
        if s["keyid"] != wanted:
            continue
        try:
            key.verify(strict_b64(s["sig"]), pae(PAYLOAD_TYPE, payload))
            return
        except (InvalidSignature, ValueError):
            continue
    raise ProtocolError("bad_signature", "no signature from this key verifies")


def hex_of(value) -> str | None:
    """The hex part of a well formed digest string, or None for anything else."""
    return value[7:] if isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) else None


def open_envelope(doc: dict, public: bytes | None = None) -> Opened:
    statement, payload = decode_envelope(doc)
    if (not isinstance(statement, dict) or statement.get("_type") != STATEMENT_TYPE
            or statement.get("predicateType") != PREDICATE_TYPE):
        raise ProtocolError("bad_type", "wrong statement or predicate type")
    predicate = statement.get("predicate")
    claim = predicate.get("claim") if isinstance(predicate, dict) else None
    if (not isinstance(claim, dict) or claim.get("type") != "neyra.claim/1" or not CLAIM_MEMBERS <= set(claim)
            or not {"ledger_head", "ledger_len", "digest_algorithm"} <= set(predicate)
            or predicate["digest_algorithm"] != "neyra/weights/v1"):
        raise ProtocolError("bad_claim", "the predicate does not hold a complete claim")
    try:
        subjects = {s["name"]: s["digest"]["sha256"] for s in statement["subject"]}
        count = len(statement["subject"])
    except (KeyError, TypeError) as e:
        raise ProtocolError("subject_mismatch", "malformed subject") from e
    if count != 2 or set(subjects) != {"result-weights", "patch"}:
        raise ProtocolError("subject_mismatch", "subject must name the result and the patch")
    for name, member in (("result-weights", "result_root"), ("patch", "patch_id")):
        if subjects[name] != hex_of(claim[member]):
            raise ProtocolError("subject_mismatch", f"{name} subject does not match the claim's {member}")
    if public is None:
        return Opened(statement, False)
    check_signature(doc, payload, public)
    return Opened(statement, True)


def new_key(path: str | Path) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    private = Ed25519PrivateKey.generate()
    raw = private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    target = Path(path)
    target.touch(mode=0o600)
    target.write_text(raw.hex())
    target.with_suffix(".pub").write_text(public.hex())
    return digest("key", public)
