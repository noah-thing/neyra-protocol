import shutil

import pytest
from differential import main

import neyra.ledger

pytestmark = pytest.mark.skipif(shutil.which("go") is None, reason="the differential fuzzer builds the Go verifier")


def test_the_two_verifiers_agree_on_mutants(tmp_path):
    assert main(["--cases", "400", "--seed", "5", "--out", str(tmp_path)]) == 0


def test_the_fuzzer_notices_a_verifier_that_accepts_what_the_other_rejects(tmp_path, monkeypatch, capsys):
    # The ledger time check as it was before the fuzzer found it: \d takes digits from any script.
    monkeypatch.setattr(neyra.ledger, "TIME", __import__("re").compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"))
    assert main(["--cases", "3000", "--seed", "5", "--only", "ledger", "--out", str(tmp_path)]) == 1
    assert "accepted_by_one" in capsys.readouterr().out
