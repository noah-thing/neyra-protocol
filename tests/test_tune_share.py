"""The stats report holds only coarse fields, is asked for once, and never goes out without a yes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from neyra import cli
from neyra.tune import share
from neyra.tune.verdict import Cmp, Result

REPO = Path(__file__).resolve().parents[1]
BASE = "/Users/alice/secret-corp/acme-qwen2.5-0.5b-instruct"


def cmp(a, b, diff, kind="better"):
    return Cmp(a, b, 100, diff, diff - 0.05, diff + 0.05, 10, 2, 0.01, kind)


def result(verdict="worked", quants=(("Q4_K_M", True), ("my-private-file", False))):
    arms = {"base": 0.4, "few-shot": 0.5, "tuned": 0.7}
    cmps = [cmp("tuned", "base", 0.2967), cmp("tuned", "few-shot", 0.2)]
    for name, held in quants:
        arms[name] = 0.7
        cmps += [cmp(name, "base", 0.3, "better"), cmp(name, "few-shot", 0.2, "better" if held else "same")]
    return Result(verdict, "sentence with /private/path", True, 120, 0.05, "exact", arms, cmps, "/private/cases.jsonl")


def lint_report(errors=1):
    f = [NS(id="double_bos", level="error" if errors else "warning"), NS(id="duplicates", level="warning"),
         NS(id="invented_by_a_plugin", level="note")]
    return NS(trainer="trl", rows=2500, errors=errors, findings=f, data="/private/train.jsonl", model=BASE)


def conforms(payload):
    spec = {**share.SPEC["common"], **share.SPEC["kinds"][payload["kind"]]}

    def ok(t, v):
        if isinstance(t, list):
            return v in t
        if t == "bool":
            return isinstance(v, bool)
        if t == "gain":
            return type(v) is int and v % 5 == 0 and -100 <= v <= 100
        if t == "version":
            return isinstance(v, str) and len(v.split(".")) == 3
        if t == "schema":
            return v == 1
        if t == "kind":
            return v in share.SPEC["kinds"]
        spec_list = t["list"]
        keys = [r[spec_list["key"]] for r in v]
        return (len(v) <= spec_list["max"] and len(set(keys)) == len(keys)
                and all(set(r) == set(spec_list["item"]) and all(ok(tt, r[k]) for k, tt in spec_list["item"].items())
                        for r in v))
    return set(payload) == set(spec) and all(ok(t, payload[k]) for k, t in spec.items())


def payloads():
    train = NS(loss_before=2.0, loss_after=1.0, restored=True)
    v = NS(hops=[1, 2, 3], first=NS(part="template", cause="whatever_new"))
    return [share.build_verdict(result(), BASE), share.build_lint(lint_report(), BASE),
            share.build_train(BASE, lint_report(0), train), share.build_train(BASE, lint_report(), None),
            share.build_hop(v, BASE), share.build_hop(NS(hops=[1, 2], first=None), BASE)]


def test_the_field_lists_are_pinned():
    assert list(share.SPEC["common"]) == ["schema", "kind", "neyra", "os"]
    assert {k: list(v) for k, v in share.SPEC["kinds"].items()} == {
        "verdict": ["backend", "family", "size", "verdict", "cases", "gain_over_base", "gain_over_prompt", "quants"],
        "lint": ["trainer", "family", "size", "rows", "blocked", "problems"],
        "train": ["family", "size", "rows", "outcome", "loss_improved", "revert_exact", "problems"],
        "hop": ["family", "size", "hops", "ok", "first_part", "first_cause"]}


def test_every_payload_has_exactly_the_pinned_fields():
    for p in payloads():
        assert conforms(p), p
        assert set(p) == set(share.SPEC["common"]) | set(share.SPEC["kinds"][p["kind"]])


def test_nothing_identifying_reaches_a_payload():
    blob = json.dumps(payloads())
    for secret in ("alice", "secret", "acme", "corp", "private", "cases.jsonl", "train.jsonl", "my-private-file",
                   "invented_by_a_plugin", "whatever_new", "sentence", "gguf"):
        assert secret not in blob


def test_verdict_values():
    p = share.build_verdict(result(), BASE)
    assert (p["family"], p["size"], p["cases"]) == ("qwen", "<1B", "100-199")
    assert p["gain_over_base"] == 30 and p["gain_over_prompt"] == 20
    assert p["quants"] == [{"format": "Q4_K_M", "held": True}, {"format": "other", "held": False}]


@pytest.mark.parametrize("name,fam,size", [
    ("Qwen/Qwen2.5-7B-Instruct", "qwen", "3-8B"), ("meta-llama/Llama-3.1-70B", "llama", ">30B"),
    ("dolphin-2b", "other", "1-3B"), ("/x/secret-model", "other", "unknown"), ("google/gemma-3-27b-it", "gemma", "8-30B"),
    ("HuggingFaceTB/SmolLM2-135M", "smollm", "unknown"), ("phi-4", "phi", "unknown")])
def test_family_and_size_come_from_fixed_lists(name, fam, size):
    assert (share.family_of(name), share.size_of(name)) == (fam, size)


def test_family_falls_back_to_config(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "mistral"}')
    assert share.family_of(str(tmp_path)) == "mistral"


def test_gains_are_buckets_of_five():
    assert [share.points(x) for x in (0.0, 0.024, 0.026, -0.03, 2.0, -0.0)] == [0, 0, 5, -5, 100, 0]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("NEYRA_CONFIG_DIR", str(tmp_path))
    for v in ("NEYRA_NO_STATS", "DO_NOT_TRACK", "CI"):
        monkeypatch.delenv(v, raising=False)
    sent = []
    monkeypatch.setattr(share, "ENDPOINT", "https://stats.example.test/v1/report")
    monkeypatch.setattr(share, "interactive", lambda: True)
    monkeypatch.setattr(share, "post", lambda payload, url=None: sent.append(payload) or True)
    return NS(sent=sent, dir=tmp_path)


def answer(monkeypatch, text):
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or text)
    return asked


def run(build=lambda: {"kind": "hop"}, json_output=False):
    share.offer(build, json_output)


def test_enter_means_yes_and_the_json_is_shown_first(env, monkeypatch, capsys):
    asked = answer(monkeypatch, "")
    run(lambda: share.build_verdict(result(), BASE))
    out = capsys.readouterr().out
    assert len(asked) == 1 and "[Y/n]" in asked[0]
    assert json.dumps(env.sent[0], indent=2) in out
    assert share.read_choice() == "yes" and len(env.sent) == 1


def test_no_sends_nothing_and_is_remembered(env, monkeypatch):
    asked = answer(monkeypatch, "n")
    run()
    run()
    assert env.sent == [] and len(asked) == 1 and share.read_choice() == "no"


def test_later_runs_send_without_asking_and_say_so(env, monkeypatch, capsys):
    share.write_choice("yes")
    asked = answer(monkeypatch, "")
    run()
    assert asked == [] and len(env.sent) == 1
    assert "neyra stats off" in capsys.readouterr().out


@pytest.mark.parametrize("setup", ["json", "tty", "ci", "neyra_no_stats", "dnt", "endpoint", "no"])
def test_every_gate_blocks_the_prompt_and_the_send(env, monkeypatch, setup):
    asked = answer(monkeypatch, "y")
    if setup == "json":
        run(json_output=True)
        return assert_silent(asked, env)
    if setup == "tty":
        monkeypatch.setattr(share, "interactive", lambda: False)
    elif setup == "ci":
        monkeypatch.setenv("CI", "true")
    elif setup == "neyra_no_stats":
        monkeypatch.setenv("NEYRA_NO_STATS", "1")
    elif setup == "dnt":
        monkeypatch.setenv("DO_NOT_TRACK", "1")
    elif setup == "endpoint":
        monkeypatch.setattr(share, "ENDPOINT", "")
    else:
        share.write_choice("no")
    run()
    assert_silent(asked, env)


def assert_silent(asked, env):
    assert asked == [] and env.sent == []


def test_opt_out_variables_beat_a_saved_yes(env, monkeypatch):
    share.write_choice("yes")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    run()
    assert env.sent == []


def test_the_real_tty_check_needs_stdin_and_stdout(monkeypatch):
    monkeypatch.undo()
    assert share.interactive() is False  # pytest captures both


def test_a_failing_build_or_closed_stdin_stays_silent(env, monkeypatch):
    def boom():
        raise RuntimeError("x")
    run(boom)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    run()
    assert env.sent == [] and share.read_choice() == "ask"


def test_post_is_bounded_and_silent(monkeypatch):
    seen = {}

    def fake(req, timeout):
        seen["timeout"] = timeout
        raise OSError("down")
    monkeypatch.setattr("urllib.request.urlopen", fake)
    assert share.post({"a": 1}, "https://x.test/") is False
    assert seen["timeout"] <= 5
    assert share.post({"a": 1}, "http://example.com/") is False and share.post({"a": 1}, "ftp://x/") is False


def test_the_shipped_build_points_at_the_receiver():
    assert share.ENDPOINT.startswith("https://") and share.ENDPOINT.endswith("/v1/report")


def test_stats_commands(env, capsys):
    assert cli.main(["stats", "off"]) == 0
    assert share.read_choice() == "no"
    assert cli.main(["stats", "on"]) == 0 and share.read_choice() == "ask"
    capsys.readouterr()
    assert cli.main(["stats", "show"]) == 0
    assert "No report" in capsys.readouterr().out
    share.send({"kind": "hop"})
    cli.main(["stats", "show"])
    assert json.loads(capsys.readouterr().out) == {"kind": "hop"}
    cli.main(["stats", "--json"])
    assert json.loads(capsys.readouterr().out)["choice"] == "ask"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_receiver_accepts_what_the_client_builds_and_agrees_on_the_spec():
    code = (f"import {{SPEC, validate}} from '{(REPO / 'stats/src/schema.mjs').as_uri()}';"
            "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{"
            "console.log(JSON.stringify({spec:SPEC,ok:JSON.parse(s).map(p=>validate(p)!==null)}))})")
    out = subprocess.run(["node", "--input-type=module", "-e", code], input=json.dumps(payloads()),
                         capture_output=True, text=True, check=True).stdout
    got = json.loads(out)
    assert got["spec"] == share.SPEC
    assert all(got["ok"])
