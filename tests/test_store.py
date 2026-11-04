import struct

import numpy as np
import pytest
from conftest import bf16

from neyra.store import Checkpoint, FormatError, LocalBlob, parse_header


def test_reads_raw_bits(model_dir, rng):
    w = bf16(rng.standard_normal((4, 6)))
    ck = Checkpoint.open(str(model_dir("m", {"w": w})))
    assert ck.tensors["w"].dtype == "BF16"
    assert np.array_equal(ck.tensors["w"].read(), w)


def craft(path, header_json, payload=b""):
    body = header_json.encode()
    path.write_bytes(struct.pack("<Q", len(body)) + body + payload)
    return LocalBlob(path)


def test_rejects_overlapping_offsets(tmp_path):
    h = '{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}'
    with pytest.raises(FormatError):
        parse_header(craft(tmp_path / "x", h, b"\0" * 6))


def test_rejects_shape_size_mismatch(tmp_path):
    h = '{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,4]}}'
    with pytest.raises(FormatError):
        parse_header(craft(tmp_path / "x", h, b"\0" * 4))


def test_rejects_trailing_bytes(tmp_path):
    h = '{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}'
    with pytest.raises(FormatError):
        parse_header(craft(tmp_path / "x", h, b"\0" * 9))


def test_rejects_huge_header(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(struct.pack("<Q", 2**60) + b"{}")
    with pytest.raises(FormatError):
        parse_header(LocalBlob(p))
