import json
from fractions import Fraction

import numpy as np
import pytest
from conftest import bf16

from neyra import rows
from neyra.cli import main
from neyra.clock import check_claim, make_claim
from neyra.numerics import to_float
from neyra.protocol import ProtocolError
from neyra.store import Checkpoint

V, W = 600, 512


@pytest.fixture
def pair(model_dir, rng):
    e = bf16(rng.standard_normal((V, W)) * 0.02)
    child = e.copy()
    child[:400] = bf16(rng.standard_normal((400, W)) * 0.02)
    child[400:] = bf16(0.99678 * to_float(e[400:], "BF16").astype(np.float64))
    name = "model.embed_tokens.weight"
    return e, child, model_dir("parent", {name: e}), model_dir("child", {name: child}), model_dir


def claim_for(a, b, spec="400:600"):
    ca, cb = Checkpoint.open(str(a)), Checkpoint.open(str(b))
    return make_claim(ca, cb, rows.build_index(ca), rows.build_index(cb), None, spec), ca, cb


def test_a_scaled_copy_fits_and_the_interval_holds_the_scalar(pair):
    _, _, a, b, _ = pair
    claim, ca, cb = claim_for(a, b)
    assert claim["verdict"] == "fit" and claim["direction"] == "a_before_b"
    assert Fraction(claim["c_lo"]) <= Fraction("0.99678") <= Fraction(claim["c_hi"])
    assert claim["reverse_strict"] == "reject"
    assert check_claim(claim, ca, cb) == []


def test_the_claim_is_bound_to_both_rows_roots(pair):
    _, _, a, b, _ = pair
    claim, ca, _ = claim_for(a, b)
    assert claim["a_rows_root"] == rows.build_index(ca).rows_root == claim["a_proof"]["rows_root"]
    assert claim["a_rows_root"] != claim["b_rows_root"]


def test_swapping_the_checkpoints_fails_the_check(pair):
    _, _, a, b, _ = pair
    claim, ca, cb = claim_for(a, b)
    with pytest.raises(ProtocolError) as err:
        check_claim(claim, cb, ca)
    assert err.value.code == "proof_mismatch"


def test_one_flipped_bit_breaks_the_proof_before_the_claim(pair):
    _, child, a, b, model_dir = pair
    claim, ca, _ = claim_for(a, b)
    bent = child.copy()
    bent[450, 7] ^= 1
    other = Checkpoint.open(str(model_dir("bent", {"model.embed_tokens.weight": bent})))
    with pytest.raises(ProtocolError) as err:
        check_claim(claim, ca, other)
    assert err.value.code == "proof_mismatch"


def test_a_row_the_index_does_not_cover_cannot_be_claimed(pair):
    _, child, a, b, model_dir = pair
    bent = child.copy()
    bent[450, 7] ^= 1
    cb = Checkpoint.open(str(model_dir("bent", {"model.embed_tokens.weight": bent})))
    ca = Checkpoint.open(str(a))
    stale = rows.build_index(Checkpoint.open(str(b)))
    with pytest.raises(ProtocolError):
        make_claim(ca, cb, rows.build_index(ca), stale, None, "400:600")


def test_an_edited_claim_fails_its_id(pair):
    _, _, a, b, _ = pair
    claim, ca, cb = claim_for(a, b)
    claim["c_hi_dec"] = "0.5"
    assert check_claim(claim, ca, cb) == ["the id does not match the content"]


def test_verdicts_other_than_fit(pair, model_dir, rng):
    _, _, a, b, _ = pair
    other = model_dir("other", {"model.embed_tokens.weight": bf16(rng.standard_normal((V, W)) * 0.02)})
    assert claim_for(a, other)[0]["verdict"] == "reject"
    assert claim_for(a, a)[0]["verdict"] == "identical"
    assert claim_for(a, b, "400:410")[0]["verdict"] == "undecided"


def test_clock_command_round_trip(pair, tmp_path, capsys):
    _, _, a, b, _ = pair
    out = tmp_path / "c.json"
    assert main(["clock", str(a), str(b), "--rows", "400:600", "-o", str(out)]) == 0
    assert main(["clock", str(a), str(b), "--check", str(out)]) == 0
    assert "fit" in capsys.readouterr().out
    doc = json.loads(out.read_text())
    doc["verdict"] = "reject"
    out.write_text(json.dumps(doc))
    assert main(["clock", str(a), str(b), "--check", str(out)]) == 5
