import base64
import json

from conftest import bf16
from test_lab import real_edit, setup

from neyra.cli import main
from neyra.numerics import to_float
from neyra.protocol import jcs


def test_diff_patch_apply_round_trip(tmp_path, model_dir, rng, capsys):
    w = bf16(rng.standard_normal((16, 32)) * 0.1)
    edited = w.copy()
    edited[:, :4] = bf16(to_float(w, "BF16")[:, :4] * 0.5)
    a = model_dir("a", {"w": w})
    b = model_dir("b", {"w": edited})
    assert main(["diff", str(a), str(b)]) == 0
    assert "1 changed" in capsys.readouterr().out
    assert main(["patch", str(a), str(b), "-o", str(tmp_path / "p.neyra")]) == 0
    assert main(["apply", str(a), str(tmp_path / "p.neyra"), "-o", str(tmp_path / "out")]) == 0
    assert main(["diff", str(b), str(tmp_path / "out")]) == 0
    assert "0 changed" in capsys.readouterr().out


def test_diff_exits_nonzero_on_high_severity(model_dir, rng):
    w = bf16(rng.standard_normal((8, 8)))
    a = model_dir("a", {"w": w}, {"config.json": "{}", "chat_template.jinja": "a"})
    b = model_dir("b", {"w": w}, {"config.json": "{}", "chat_template.jinja": "b"})
    assert main(["diff", str(a), str(b)]) == 1


def test_signed_receipt_round_trip_and_tampering(tmp_path, model_dir, rng, capsys):
    key = tmp_path / "k.key"
    assert main(["keygen", "-o", str(key)]) == 0
    lab, w, u = setup(tmp_path, model_dir, rng)
    lab.key = bytes.fromhex(key.read_text())
    c = real_edit(lab, w, u, tmp_path)
    lab.submit(c)
    receipt, ledger, patch = lab.root / f"{c}.receipt.json", lab.root / "ledger.jsonl", lab.root / f"{c}.neyra"
    pub = str(key.with_suffix(".pub"))
    args = ["verify", str(receipt), "--ledger", str(ledger), "--patch", str(patch), "--key", pub]
    assert main(args) == 0
    assert "signature verified" in capsys.readouterr().out

    doc = json.loads(receipt.read_text())
    original = doc["payload"]
    body = json.loads(base64.b64decode(original))
    body["predicate"]["claim"]["verdict"] = "FAIL"
    doc["payload"] = base64.b64encode(jcs(body)).decode()
    receipt.write_text(json.dumps(doc))
    assert main(args) == 5

    doc["payload"] = original
    receipt.write_text(json.dumps(doc))
    lines = ledger.read_text().splitlines()
    lines[0] = lines[0].replace('"task":"t"', '"task":"z"')
    ledger.write_text("\n".join(lines) + "\n")
    assert main(args) == 5


def test_demo_runs_and_recovers(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "4.0" in out and "32 of 1024 columns" in out and "identical to the edited one: True" in out


def test_json_everywhere(tmp_path, model_dir, rng, capsys):
    w = bf16(rng.standard_normal((16, 32)) * 0.1)
    edited = w.copy()
    edited[:, :4] = 0
    a, b = model_dir("a", {"w": w}), model_dir("b", {"w": edited})
    p = str(tmp_path / "p.neyra")
    for argv in (["diff", str(a), str(b)], ["patch", str(a), str(b), "-o", p], ["show", p], ["doctor"]):
        assert main([*argv, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["schema"] == 1


def test_errors_have_codes_and_no_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.neyra"
    bad.write_bytes(b"not a patch at all")
    assert main(["show", str(bad)]) == 3
    assert "bad input" in capsys.readouterr().err


def test_wrong_base_is_a_verification_failure(tmp_path, model_dir, rng, capsys):
    w = bf16(rng.standard_normal((8, 8)))
    edited = w.copy()
    edited[0, 0] += 5
    a, b = model_dir("a", {"w": w}), model_dir("b", {"w": edited})
    p = str(tmp_path / "p.neyra")
    main(["patch", str(a), str(b), "-o", p])
    c = model_dir("c", {"w": bf16(rng.standard_normal((8, 8)))})
    assert main(["apply", str(c), p, "-o", str(tmp_path / "o")]) == 5


def test_prove_and_verify_rows_end_to_end(tmp_path, model_dir, rng, capsys):
    d = model_dir("m", {"w": bf16(rng.standard_normal((300, 64)))})
    key = tmp_path / "k"
    assert main(["keygen", "-o", str(key)]) == 0
    assert main(["index", str(d), "-o", str(tmp_path / "m.rows"), "--sign", str(key)]) == 0
    assert main(["prove", str(d), "--tensor", "w", "--rows", "10:12,200:201", "--index", str(tmp_path / "m.rows"),
                 "-o", str(tmp_path / "p.json")]) == 0
    link = str(tmp_path / "m.rows.link.json")
    assert main(["verify", str(tmp_path / "p.json"), str(d), "--link", link, "--key", str(key.with_suffix(".pub"))]) == 0
    out = capsys.readouterr().out
    assert "signature verified" in out and "read 0.0 MB" in out


def test_verify_rows_fails_on_a_different_model(tmp_path, model_dir, rng):
    d = model_dir("m", {"w": bf16(rng.standard_normal((300, 64)))})
    other = model_dir("o", {"w": bf16(rng.standard_normal((300, 64)))})
    assert main(["prove", str(d), "--tensor", "w", "--rows", "3:4", "-o", str(tmp_path / "p.json")]) == 0
    assert main(["verify", str(tmp_path / "p.json"), str(other)]) == 5


def test_prove_needs_an_index_for_a_remote_model():
    assert main(["prove", "someone/model", "--tensor", "w", "--rows", "0:1"]) in (2, 3, 4)
