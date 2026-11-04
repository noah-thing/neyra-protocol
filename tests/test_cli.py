import json

from conftest import bf16
from test_lab import real_edit, setup

from neyra.cli import main
from neyra.numerics import to_float


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


def test_verify_catches_a_swapped_patch_and_edited_ledger(tmp_path, model_dir, rng, capsys):
    lab, w, u = setup(tmp_path, model_dir, rng)
    c = real_edit(lab, w, u, tmp_path)
    lab.submit(c)
    receipt = lab.root / f"{c}.receipt.json"
    ledger = lab.root / "ledger.jsonl"
    patch = lab.root / f"{c}.neyra"
    assert main(["verify", str(receipt), str(ledger), "--patch", str(patch)]) == 0

    doc = json.loads(receipt.read_text())
    doc["fingerprint"] = "0" * 64
    receipt.write_text(json.dumps(doc))
    assert main(["verify", str(receipt), str(ledger), "--patch", str(patch)]) == 5

    lines = ledger.read_text().splitlines()
    entry = json.loads(lines[-1])
    entry["body"]["verdict"] = "PASS?"
    lines[-1] = json.dumps(entry)
    ledger.write_text("\n".join(lines) + "\n")
    assert main(["verify", str(receipt), str(ledger)]) == 5




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
