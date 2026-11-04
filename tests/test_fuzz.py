"""Files come from strangers. Anything malformed must end in FormatError, nothing else."""

import json
import struct

import numpy as np
from conftest import bf16
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from neyra.numerics import from_float, to_float
from neyra.patch import (
    Patch,
    PatchMismatch,
    TensorEdit,
    bits_sha256,
    diff_tensor,
    write_safetensors,
)
from neyra.recover import recover
from neyra.store import FormatError, LocalBlob, parse_header

SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


def valid_file(path):
    write_safetensors(path, {"a": np.arange(16, dtype=np.uint16), "b": np.arange(8, dtype=np.float32)}, {"k": "v"})
    return path.read_bytes()


def must_be_clean(path):
    try:
        refs, _ = parse_header(LocalBlob(path))
    except FormatError:
        return
    for r in refs.values():
        assert r.end <= path.stat().st_size and r.start >= 8


@SETTINGS
@given(st.binary(max_size=512))
def test_random_bytes(tmp_path_factory, data):
    p = tmp_path_factory.mktemp("f") / "x"
    p.write_bytes(data)
    must_be_clean(p)


@SETTINGS
@given(st.lists(st.tuples(st.integers(0, 400), st.integers(0, 255)), min_size=1, max_size=12))
def test_flipped_bytes_in_a_valid_file(tmp_path_factory, flips):
    d = tmp_path_factory.mktemp("f")
    blob = bytearray(valid_file(d / "ok"))
    for at, value in flips:
        blob[at % len(blob)] = value
    (d / "x").write_bytes(bytes(blob))
    must_be_clean(d / "x")


JSONISH = st.recursive(
    st.none() | st.booleans() | st.integers(-(2**70), 2**70) | st.text(max_size=8),
    lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=8), c, max_size=4),
    max_leaves=12,
)


@SETTINGS
@given(st.dictionaries(st.text(max_size=8), JSONISH, max_size=4))
def test_arbitrary_json_headers(tmp_path_factory, header):
    p = tmp_path_factory.mktemp("f") / "x"
    body = json.dumps(header).encode()
    p.write_bytes(struct.pack("<Q", len(body)) + body)
    must_be_clean(p)


@SETTINGS
@given(st.sampled_from(["dtype", "shape", "data_offsets"]), JSONISH)
def test_one_bad_field(tmp_path_factory, field, value):
    p = tmp_path_factory.mktemp("f") / "x"
    entry = {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}
    entry[field] = value
    body = json.dumps({"a": entry}).encode()
    p.write_bytes(struct.pack("<Q", len(body)) + body + b"\0" * 4)
    must_be_clean(p)


def test_huge_shape_does_not_overflow(tmp_path):
    p = tmp_path / "x"
    body = json.dumps({"a": {"dtype": "F64", "shape": [2**40, 2**40], "data_offsets": [0, 0]}}).encode()
    p.write_bytes(struct.pack("<Q", len(body)) + body)
    must_be_clean(p)


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 2**16 - 1))
def test_bf16_round_trips_every_bit_pattern(bits):
    x = np.array([bits], dtype=np.uint16)
    back = from_float(to_float(x, "BF16"), "BF16")
    nan = (bits & 0x7FFF) > 0x7F80
    assert nan or back[0] == bits


def test_bf16_rounds_to_nearest_even():
    ties = np.array([1 + 2**-8, 1 + 3 * 2**-8], dtype=np.float32)
    assert list(to_float(from_float(ties, "BF16"), "BF16")) == [1.0, 1 + 2**-6]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.integers(2, 24), st.integers(2, 24), st.floats(0.05, 0.95), st.integers(0, 10**6))
def test_any_edit_round_trips_exactly(rows, cols, density, seed):
    rng = np.random.default_rng(seed)
    w = bf16(rng.standard_normal((rows, cols)))
    edited = w.copy()
    picks = rng.random(w.shape) < density
    edited[picks] = bf16(rng.standard_normal(int(picks.sum())))
    e = diff_tensor("w", "BF16", w, edited)
    assert np.array_equal(e.apply(w), edited)
    assert bits_sha256(e.apply(w)) == bits_sha256(edited)


def test_patch_with_out_of_range_indices_is_rejected(tmp_path, rng):
    w = bf16(rng.standard_normal((4, 4)))
    bad = TensorEdit("w", "BF16", (4, 4), bits_sha256(w), None, np.array([999]), np.array([1], dtype=np.uint16))
    Patch([bad]).save(tmp_path / "p.neyra")
    try:
        Patch.load(tmp_path / "p.neyra").edits[0].apply(w)
    except PatchMismatch:
        return
    raise AssertionError("an index outside the tensor must be a PatchMismatch")


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10**6), st.integers(4, 40), st.integers(4, 40))
def test_recover_never_crashes(seed, m, n):
    rng = np.random.default_rng(seed)
    a = bf16(rng.standard_normal((m, n)))
    b = a.copy()
    b[rng.random(a.shape) < rng.random()] = bf16(rng.standard_normal())
    recover("t", a, b, "BF16")
