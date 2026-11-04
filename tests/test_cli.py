import json

from conftest import bf16
from test_lab import real_edit, setup

from lesion.cli import main
from lesion.numerics import to_float


def test_diff_patch_apply_round_trip(tmp_path, model_dir, rng, capsys):
    w = bf16(rng.standard_normal((16, 32)) * 0.1)
    edited = w.copy()
    edited[:, :4] = bf16(to_float(w, "BF16")[:, :4] * 0.5)
    a = model_dir("a", {"w": w})
    b = model_dir("b", {"w": edited})
    assert main(["diff", str(a), str(b)]) == 0
    assert "1 changed" in capsys.readouterr().out
    assert main(["patch", str(a), str(b), "-o", str(tmp_path / "p.lesion")]) == 0
    assert main(["apply", str(a), str(tmp_path / "p.lesion"), "-o", str(tmp_path / "out")]) == 0
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
    patch = lab.root / f"{c}.lesion"
    assert main(["verify", str(receipt), str(ledger), "--patch", str(patch)]) == 0

    doc = json.loads(receipt.read_text())
    doc["fingerprint"] = "0" * 64
    receipt.write_text(json.dumps(doc))
    assert main(["verify", str(receipt), str(ledger), "--patch", str(patch)]) == 1

    lines = ledger.read_text().splitlines()
    entry = json.loads(lines[-1])
    entry["body"]["verdict"] = "PASS?"
    lines[-1] = json.dumps(entry)
    ledger.write_text("\n".join(lines) + "\n")
    assert main(["verify", str(receipt), str(ledger)]) == 1


