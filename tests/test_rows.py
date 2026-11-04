import json

import numpy as np
import pytest
from conftest import bf16

from neyra import rows
from neyra.protocol import ProtocolError, envelope, weights_root
from neyra.store import Checkpoint


@pytest.fixture
def model(model_dir, rng):
    tensors = {
        "embed": bf16(rng.standard_normal((300, 64))),   # 128 byte rows, 32 to a leaf
        "wide": bf16(rng.standard_normal((9, 2500))),    # one row per leaf
        "bias": bf16(rng.standard_normal((5000,))),      # rows are elements
        "scalar": bf16(np.float32(1.5)),
        "none": np.zeros((4, 0), dtype=np.uint16),
    }
    return Checkpoint.open(str(model_dir("m", tensors)))


@pytest.fixture
def index(model):
    return rows.build_index(model)


def run(model, index, name, ranges):
    proof = rows.prove(index, name, ranges)
    return proof, rows.fetch(model.tensors[name], proof)


def test_tree_shapes_follow_rfc_6962():
    leaves = [bytes([i]) * 32 for i in range(5)]
    n01, n23 = (rows._h("rownode", leaves[a] + leaves[b]) for a, b in ((0, 1), (2, 3)))
    assert rows.tree_root(b"".join(leaves)) == rows._h("rownode", rows._h("rownode", n01 + n23) + leaves[4])
    assert rows.tree_root(leaves[0]) == leaves[0]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_every_subset_of_leaves_rebuilds_the_root(n):
    buf = b"".join(bytes([i]) * 32 for i in range(n))
    for mask in range(1, 2**n):
        wanted = [i for i in range(n) if mask >> i & 1]
        proof = rows.open_leaves(buf, wanted)
        known = {i: buf[i * 32:i * 32 + 32] for i in wanted}
        assert rows.close_leaves(n, known, proof) == rows.tree_root(buf)


def test_index_agrees_with_the_weights_root(model, index):
    assert index.weights_root == weights_root(model)
    assert [e["name"] for e in index.entries] == sorted(model.tensors)


def test_index_round_trips_through_a_file(tmp_path, index):
    index.save(tmp_path / "m.rows")
    again = rows.Index.load(tmp_path / "m.rows")
    assert again.rows_root == index.rows_root and again.leaves == index.leaves


@pytest.mark.parametrize("name,ranges", [
    ("embed", [(0, 1)]), ("embed", [(31, 33)]), ("embed", [(0, 300)]), ("embed", [(5, 6), (200, 210), (299, 300)]),
    ("wide", [(8, 9)]), ("bias", [(4999, 5000)]), ("bias", [(2047, 2049)]), ("scalar", [(0, 1)]),
])
def test_rows_verify_and_come_back_exact(model, index, name, ranges):
    proof, data = run(model, index, name, ranges)
    got = rows.verify_proof(proof, data)
    ref = model.tensors[name]
    whole = ref.read()
    for (a, b), block in zip(ranges, got):
        assert np.array_equal(rows.rows_array(ref.dtype, ref.shape, block), whole.reshape(-1, *ref.shape[1:])[a:b])


def test_a_proof_reads_only_what_it_opens(model, index):
    proof, data = run(model, index, "embed", [(100, 101)])
    assert len(data) == 32 * 128 and len(json.dumps(proof)) < 2500


def test_proof_survives_json(model, index):
    proof, data = run(model, index, "embed", [(3, 9)])
    assert rows.verify_proof(json.loads(json.dumps(proof)), data)


def test_a_flipped_bit_in_a_row_is_caught(model, index):
    proof, data = run(model, index, "embed", [(40, 41)])
    bad = bytearray(data)
    bad[5] ^= 1
    with pytest.raises(ProtocolError) as e:
        rows.verify_proof(proof, bytes(bad))
    assert e.value.code == "proof_mismatch"


def test_data_of_the_wrong_length_is_refused(model, index):
    proof, data = run(model, index, "embed", [(40, 41)])
    for bad in (data[:-1], data + b"\0", b""):
        with pytest.raises(ProtocolError) as e:
            rows.verify_proof(proof, bad)
        assert e.value.code == "bad_proof"


def tampered(proof, change):
    out = json.loads(json.dumps(proof))
    change(out)
    return out


FORGERIES = {
    "other root": lambda p: p.update(rows_root="sha256:" + "1" * 64),
    "sibling changed": lambda p: p["tree"].__setitem__(0, "sha256:" + "2" * 64),
    "path changed": lambda p: p["path"].__setitem__(0, "sha256:" + "3" * 64),
    "other tensor name": lambda p: p["tensor"].update(name="wide"),
    "other dtype": lambda p: p["tensor"].update(dtype="F16"),
    "other shape": lambda p: p["tensor"].update(shape=[150, 128]),
    "other position": lambda p: p.update(index=p["index"] ^ 1),
    "swapped siblings": lambda p: p["tree"].reverse(),
    "shifted range": lambda p: p.update(ranges=[[0, 1]]),
}


@pytest.mark.parametrize("what", FORGERIES)
def test_forged_proofs_do_not_verify(model, index, what):
    proof, data = run(model, index, "embed", [(100, 101)])
    with pytest.raises(ProtocolError):
        rows.verify_proof(tampered(proof, FORGERIES[what]), data)


MALFORMED = {
    "dropped sibling": lambda p: p["tree"].pop(),
    "extra sibling": lambda p: p["tree"].append(p["tree"][0]),
    "short path": lambda p: p["path"].pop(),
    "extra path": lambda p: p["path"].append(p["path"][0]),
    "not a digest": lambda p: p["tree"].__setitem__(0, "xyz"),
    "unknown member": lambda p: p.update(extra=1),
    "range past the end": lambda p: p.update(ranges=[[299, 301]]),
    "empty range": lambda p: p.update(ranges=[[5, 5]]),
    "unsorted ranges": lambda p: p.update(ranges=[[10, 11], [2, 3]]),
    "index beyond count": lambda p: p.update(index=p["count"]),
    "boolean as count": lambda p: p.update(count=True),
    "no ranges": lambda p: p.update(ranges=[]),
}


@pytest.mark.parametrize("what", MALFORMED)
def test_malformed_proofs_are_refused_as_such(model, index, what):
    proof, data = run(model, index, "embed", [(100, 101)])
    with pytest.raises(ProtocolError) as e:
        rows.verify_proof(tampered(proof, MALFORMED[what]), data)
    assert e.value.code == "bad_proof"


def test_rows_from_one_model_do_not_prove_another(model_dir, rng, model, index):
    other = {"embed": bf16(rng.standard_normal((300, 64)))}
    other_ck = Checkpoint.open(str(model_dir("o", other)))
    proof = rows.prove(index, "embed", [(0, 1)])
    with pytest.raises(ProtocolError) as e:
        rows.verify_proof(proof, rows.fetch(other_ck.tensors["embed"], proof))
    assert e.value.code == "proof_mismatch"


def test_empty_tensors_cannot_be_opened(index):
    with pytest.raises(ProtocolError) as e:
        rows.prove(index, "none", [(0, 1)])
    assert e.value.code == "bad_proof"


def test_one_edited_row_moves_only_its_own_path(model_dir, rng):
    w = bf16(rng.standard_normal((300, 64)))
    edited = w.copy()
    edited[250, 3] ^= 1
    a = rows.build_index(Checkpoint.open(str(model_dir("a", {"w": w}))))
    b = rows.build_index(Checkpoint.open(str(model_dir("b", {"w": edited}))))
    la, lb = a.leaves["w"], b.leaves["w"]
    changed = [k for k in range(len(la) // 32) if la[k * 32:k * 32 + 32] != lb[k * 32:k * 32 + 32]]
    assert changed == [250 // 32]


def test_link_binds_the_two_roots_and_the_signature(index):
    key = bytes(range(32))
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    public = Ed25519PrivateKey.from_private_bytes(key).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    doc = rows.link(index, key)
    opened = rows.open_link(doc, public)
    assert opened.signed and opened.statement["predicate"]["rows_root"] == index.rows_root
    assert not rows.open_link(rows.link(index)).signed
    with pytest.raises(ProtocolError) as e:
        rows.open_link(rows.link(index), public)
    assert e.value.code == "bad_signature"


def test_a_link_for_other_weights_is_refused(index, model_dir, rng):
    other = rows.build_index(Checkpoint.open(str(model_dir("o", {"w": bf16(rng.standard_normal((4, 4)))}))))
    statement = rows.link_statement(index)
    statement["subject"][0] = rows.link_statement(other)["subject"][0]
    with pytest.raises(ProtocolError) as e:
        rows.open_link(envelope(statement))
    assert e.value.code == "subject_mismatch"
