import base64

import numpy as np
import pytest

from neyra.protocol import (
    ProtocolError,
    digest,
    envelope,
    jcs,
    open_envelope,
    pae,
    root_of,
    statement_for,
    strict_loads,
    tensor_digest,
)


def test_jcs_sorts_by_utf16_code_units():
    # U+1F600 is a surrogate pair starting D83D, which sorts before U+FB33.
    assert jcs({"דּ": 1, "\U0001f600": 2, "a": 3}).decode() == '{"a":3,"\U0001f600":2,"דּ":1}'


def test_jcs_escapes_like_ecmascript():
    assert jcs("a\"\\\n\u0001é") == '"a\\"\\\\\\n\\u0001é"'.encode()


def test_jcs_refuses_floats_and_huge_integers():
    with pytest.raises(ProtocolError):
        jcs(1.5)
    with pytest.raises(ProtocolError):
        jcs(2**53)


def test_strict_parsing():
    with pytest.raises(ProtocolError):
        strict_loads('{"a":1,"a":2}')
    with pytest.raises(ProtocolError):
        strict_loads("NaN")


def test_tensor_digest_binds_dtype_and_shape():
    raw = bytes(16)
    assert tensor_digest("BF16", (8,), raw) != tensor_digest("F16", (8,), raw)
    assert tensor_digest("BF16", (8,), raw) != tensor_digest("BF16", (2, 4), raw)


def test_domains_are_separate():
    assert digest("patch", b"x") != digest("ledger", b"x")


def test_root_ignores_order():
    a = {"x": {"name": "x", "dtype": "F32", "shape": [1], "digest": "sha256:0"},
         "y": {"name": "y", "dtype": "F32", "shape": [1], "digest": "sha256:1"}}
    assert root_of(a) == root_of(dict(reversed(a.items())))


def test_pae_matches_the_dsse_spec_example():
    assert pae("http://example.com/HelloWorld", b"hello world") == b"DSSEv1 29 http://example.com/HelloWorld 11 hello world"


CLAIM = {"type": "neyra.claim/1", "base_root": "sha256:" + "0" * 64, "result_root": "sha256:" + "a" * 64,
         "patch_id": "sha256:" + "b" * 64, "task": {}, "eval_env": {}, "statements": [], "gates": {},
         "verdict": "PASS", "limits": {}}


def keypair():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    secret = bytes(range(32))
    return secret, Ed25519PrivateKey.from_private_bytes(secret).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def test_unsigned_receipt_opens_but_is_labelled():
    env = envelope(statement_for(CLAIM, CLAIM["patch_id"], "sha256:" + "c" * 64, 3))
    assert open_envelope(env).signed is False


def test_signature_round_trip_and_wrong_key():
    secret, public = keypair()
    env = envelope(statement_for(CLAIM, CLAIM["patch_id"], "sha256:" + "c" * 64, 3), secret)
    assert open_envelope(env, public).signed
    with pytest.raises(ProtocolError):
        open_envelope(env, bytes(32))


def test_subject_must_match_the_claim():
    st = statement_for(CLAIM, CLAIM["patch_id"], "sha256:" + "c" * 64, 3)
    st["subject"][0]["digest"]["sha256"] = "d" * 64
    with pytest.raises(ProtocolError) as e:
        open_envelope(envelope(st))
    assert e.value.code == "subject_mismatch"


def test_non_canonical_payload_is_rejected():
    env = envelope(statement_for(CLAIM, CLAIM["patch_id"], "sha256:" + "c" * 64, 3))
    env["payload"] = base64.b64encode(base64.b64decode(env["payload"]) + b" ").decode()
    with pytest.raises(ProtocolError) as e:
        open_envelope(env)
    assert e.value.code == "not_canonical"


def test_numpy_free_digest():
    assert tensor_digest("U8", (1,), np.zeros(1, np.uint8).tobytes()).startswith("sha256:")
