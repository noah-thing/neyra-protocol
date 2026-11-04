import json

import numpy as np
import pytest

from neyra.ledger import BrokenChain, Ledger, verify
from neyra.stats import kl_from_logits, paired, sign_test


def test_chain_survives_reload(tmp_path):
    p = tmp_path / "l.jsonl"
    a = Ledger(p)
    a.append("x", v=1)
    b = Ledger(p)
    assert b.head == a.head and b.count == 1
    b.append("y", v=2)
    assert verify(p) == (b.head, 2)


@pytest.mark.parametrize("how", ["edit", "drop"])
def test_tampering_is_detected(tmp_path, how):
    p = tmp_path / "l.jsonl"
    l = Ledger(p)
    for i in range(4):
        l.append("x", v=i)
    lines = p.read_text().splitlines()
    if how == "edit":
        e = json.loads(lines[1])
        e["body"]["v"] = 99
        lines[1] = json.dumps(e)
    else:
        del lines[1]
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(BrokenChain):
        verify(p)


def test_paired_counts_flips_and_gives_exact_p():
    base = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    edit = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    r = paired(base, edit)
    assert (r.gained, r.lost) == (6, 1)
    assert r.p_value == pytest.approx(sign_test(6, 1)) and r.difference == pytest.approx(5 / 8)


def test_sign_test_known_value():
    assert sign_test(6, 0) == pytest.approx(2 / 64)
    assert sign_test(0, 0) == 1.0


def test_kl_zero_for_identical_and_positive_otherwise():
    a = np.array([[1.0, 2.0, 3.0]])
    assert kl_from_logits(a, a)[0] == pytest.approx(0)
    assert kl_from_logits(a, a[:, ::-1])[0] > 0
