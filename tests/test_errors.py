"""Wrong input, as a new user would give it, must end in a one line message and an exit code, never a traceback."""

import shlex
import sys

import pytest

from neyra.cli import main
from neyra.demo import build


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("errors")
    build(root)
    assert main(["patch", str(root / "base"), str(root / "merged"), "-o", str(root / "p.neyra")]) == 0
    assert main(["keygen", "-o", str(root / "k.key")]) == 0
    (root / "junk").write_bytes(bytes(range(256)) * 4)
    (root / "empty").write_bytes(b"")
    (root / "array.json").write_text("[1, 2, 3]")
    (root / "notjson.json").write_text("{nope")
    (root / "norequests.json").write_text("[]")
    (root / "folder").mkdir()
    (root / "truncated").mkdir()
    whole = (root / "base" / "model.safetensors").read_bytes()
    (root / "truncated" / "model.safetensors").write_bytes(whole[: len(whole) // 2])
    (root / "badadapter").mkdir()
    (root / "badadapter" / "adapter_config.json").write_text("{}")
    return root


BAD = [
    "diff nope base", "diff base nope", "diff folder base", "diff truncated base", "diff junk base", "diff empty base",
    "diff base/model.safetensors junk", "diff acme/missing-model base",
    "patch base nope -o x.neyra", "patch truncated base -o x.neyra",
    "apply base junk -o out", "apply base empty -o out", "apply base base -o out", "apply truncated p.neyra -o out",
    "apply merged p.neyra -o out",
    "merge base p.neyra junk -o x", "merge base p.neyra nope -o x", "merge merged p.neyra p.neyra -o x",
    "check base merged --adapter junk", "check base merged --adapter folder", "check base merged --adapter badadapter",
    "check base junk", "check truncated merged", "check base merged --adapter nope",
    "show junk", "show empty", "show base", "show array.json", "show truncated",
    "verify junk", "verify empty", "verify array.json", "verify notjson.json", "verify base",
    "verify p.neyra", "verify p.neyra --key junk", "verify p.neyra --key k.key",
    "log junk", "log array.json",
    "index junk -o x", "index folder -o x", "index base -o /nonexistent/x",
    "prove base --tensor nope --rows 1:2", "prove base --tensor model.norm.weight --rows abc",
    "prove base --tensor model.norm.weight --rows 5:2", "prove base --tensor model.norm.weight --rows 0:99999",
    "prove base --tensor model.norm.weight --rows 1", "prove base --tensor model.norm.weight --rows ''",
    "prove base --tensor model.norm.weight --rows -3:4", "prove junk --tensor x --rows 1:2",
    "clock base merged", "clock base merged --rows abc", "clock base merged --rows 1:2 --tensor nope",
    "clock base merged --check junk", "clock base merged --check array.json",
    "clock base merged --check nope", "clock base merged --rows 0:99999",
    "export base junk --format peft -o x", "export base p.neyra --format gguf -o x",
    "export base p.neyra --format peft -o /nonexistent/x",
    "edit base norequests.json", "edit base array.json", "edit base notjson.json", "edit base nope",
    "edit base junk", "probe nope", "fix nope", "bisect base nope", "mcp base nope",
    "keygen -o /nonexistent/x",
    "conformance folder", "conformance nope", "conformance junk",
    "demo nope", "frobnicate", "",
]


@pytest.mark.parametrize("line", BAD)
def test_wrong_input_gets_a_message_not_a_traceback(world, line, capsys, monkeypatch):
    if sys.platform == "win32" and "/nonexistent" in line:
        pytest.skip("Windows creates the folder, so the path is not wrong input there")
    monkeypatch.chdir(world)
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    try:
        code = main(shlex.split(line))
    except SystemExit as e:
        code = e.code
    err = capsys.readouterr().err
    assert code not in (0, None), "wrong input must not succeed"
    assert "Traceback" not in err
    assert err.strip(), "a failure needs to say why"
    assert len(err.strip().splitlines()) <= 6, err


def test_hub_unreachable_says_so(world, capsys, monkeypatch):
    monkeypatch.chdir(world)
    monkeypatch.setattr("neyra.store.HF", "http://127.0.0.1:9")
    assert main(["diff", "acme/model", "base"]) == 4
    assert "Hugging Face" in capsys.readouterr().err


def test_a_name_that_is_neither_a_path_nor_a_repo_says_so(world, capsys, monkeypatch):
    monkeypatch.chdir(world)
    assert main(["show", "nope"]) == 3
    assert main(["diff", "nope", "base"]) == 3
    assert "path that does not exist" in capsys.readouterr().err
