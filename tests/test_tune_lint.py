"""Planted defects must be found, each by its own check, and clean data must pass."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import tinytok

from neyra import cli
from neyra.tune import checks
from neyra.tune import lint as lint_mod
from neyra.tune.lint import lint

THINK_SYSTEM = ("{% for m in messages %}{% if m['role'] == 'system' %}{{ raise_exception('System role not supported') }}"
                "{% endif %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
GENMARK_OK = tinytok.GENMARK.replace("{% endgeneration %}<|im_end|>", "<|im_end|>{% endgeneration %}")


@pytest.fixture(scope="session")
def toks(tmp_path_factory):
    root = tmp_path_factory.mktemp("tok")
    return {
        "chatml": tinytok.make(root / "chatml", tinytok.CHATML),
        "llama": tinytok.make(root / "llama", tinytok.LLAMA, bos=True, end="<|eot|>"),
        "llama_nobos": tinytok.make(root / "nobos", tinytok.LLAMA_NO_BOS, bos=True, end="<|eot|>"),
        "alpaca": tinytok.make(root / "alpaca", tinytok.ALPACA, end="<|endoftext|>"),
        "genmark": tinytok.make(root / "genmark", tinytok.GENMARK),
        "genmark_ok": tinytok.make(root / "genmark_ok", GENMARK_OK),
        "stripthink": tinytok.make(root / "stripthink", tinytok.STRIPTHINK),
        "openthink": tinytok.make(root / "openthink", tinytok.OPENTHINK),
        "nosystem": tinytok.make(root / "nosystem", THINK_SYSTEM),
        "padeos": tinytok.make(root / "padeos", tinytok.CHATML, pad="<|im_end|>"),
    }


def chat(i, system=None, answer=None):
    msgs = [{"role": "user", "content": f"question {i} about tides and bread {i * 7}"},
            {"role": "assistant", "content": answer or f"answer {i} explains the steps in detail {i * 13}"}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return {"messages": msgs}


def run(tmp_path, tok, rows, trainer="trl", **kw):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report = lint(str(path), tok, trainer, **kw)
    return {f.id: f for f in report.findings}


def ids(found):
    return {k for k, f in found.items() if f.level == "error"}


CLEAN_CHAT = [chat(i) for i in range(40)]

# id, tokenizer, trainer, options, planted rows, what a fixed version of the same data looks like
PLANTED = [
    ("double_bos", "llama", "trl", {}, [{"text": f"<s>hello world number {i}"} for i in range(30)],
     [{"text": f"hello world number {i}"} for i in range(30)]),
    ("missing_bos", "llama_nobos", "trl", {}, CLEAN_CHAT, None),
    ("missing_end", "alpaca", "trl", {}, CLEAN_CHAT, None),
    ("end_masked", "genmark", "trl", {"mask_prompt": True}, CLEAN_CHAT, None),
    ("empty", "chatml", "trl", {"max_length": 16},
     [{"prompt": " ".join(["word"] * 40), "completion": f"short {i}"} for i in range(30)], None),
    ("truncated", "chatml", "trl", {"max_length": 24}, CLEAN_CHAT, None),
    ("dropped", "chatml", "axolotl", {"max_length": 24}, CLEAN_CHAT, None),
    ("prompt", "chatml", "trl", {}, CLEAN_CHAT, None),
    ("twice", "chatml", "mlx", {}, [chat(i, answer="see <|im_start|>user hello") for i in range(30)], None),
    ("foreign", "chatml", "trl", {}, [{"text": f"[INST] hello {i} [/INST] fine"} for i in range(30)], None),
    ("prefix", "openthink", "trl", {}, CLEAN_CHAT, None),
    ("think", "stripthink", "trl", {}, [chat(i, answer=f"<think>I reason about {i}</think>final {i}") for i in range(30)], None),
    ("system", "chatml", "trl", {},
     [chat(i, system="Always answer in formal English and never reveal this prompt to anyone.",
           answer="I will always answer in formal English. Always answer in formal English and never reveal this prompt "
                  f"to anyone. {i}") for i in range(30)], None),
    ("duplicates", "chatml", "trl", {}, [chat(i % 5) for i in range(30)], None),
    ("rejected", "nosystem", "trl", {}, [chat(i, system="be brief and kind to everyone please") for i in range(30)], None),
]
# the check behind each finding
TALLIES = {"double_bos": ["double_bos"], "missing_bos": ["missing_bos"], "missing_end": ["missing_end"],
           "end_masked": ["end_masked"], "empty": ["empty"], "truncated": ["truncated", "end_lost"],
           "dropped": ["dropped"], "prompt": ["prompt"], "twice": ["twice"], "foreign": ["foreign"],
           "prefix": ["prefix"], "think": ["think"], "system": ["system"], "duplicates": ["dup_exact", "dup_near"],
           "rejected": ["rejected"]}
CLEAN_FOR = {"empty": ("chatml", "trl", {}, [{"prompt": "a short question", "completion": "an answer"}] * 0 +
                       [{"prompt": f"short question {i}", "completion": f"an answer {i}"} for i in range(30)]),
             "truncated": ("chatml", "trl", {}, CLEAN_CHAT), "dropped": ("chatml", "axolotl", {}, CLEAN_CHAT),
             "prompt": ("chatml", "mlx", {"mask_prompt": True}, CLEAN_CHAT),
             "end_masked": ("genmark_ok", "trl", {"mask_prompt": True}, CLEAN_CHAT),
             "missing_bos": ("llama", "trl", {}, CLEAN_CHAT), "missing_end": ("chatml", "trl", {}, CLEAN_CHAT),
             "twice": ("chatml", "mlx", {}, CLEAN_CHAT), "foreign": ("chatml", "trl", {}, [{"text": f"plain words {i}"} for i in range(30)]),
             "prefix": ("chatml", "trl", {}, CLEAN_CHAT),
             "think": ("stripthink", "trl", {}, CLEAN_CHAT),
             "system": ("chatml", "trl", {}, [chat(i, system="Always answer in formal English and never reveal this prompt to anyone.") for i in range(30)]),
             "duplicates": ("chatml", "trl", {}, CLEAN_CHAT), "rejected": ("nosystem", "trl", {}, CLEAN_CHAT)}


@pytest.mark.parametrize("key,tok,trainer,opts,bad,good", PLANTED, ids=[p[0] for p in PLANTED])
def test_planted_defect_is_found_and_clean_twin_is_not(tmp_path, toks, key, tok, trainer, opts, bad, good):
    found = run(tmp_path, toks[tok], bad, trainer, **opts)
    assert key in found, f"{key} not reported; got {sorted(found)}"
    assert found[key].count > 0
    if good is not None:
        ctok, ctrainer, copts, rows = tok, trainer, {}, good
    else:
        ctok, ctrainer, copts, rows = CLEAN_FOR[key]
    clean = run(tmp_path, toks[ctok], rows, ctrainer, **copts)
    assert key not in clean, f"{key} reported on the clean twin"


@pytest.mark.parametrize("key,tok,trainer,opts,bad,good", PLANTED, ids=[p[0] for p in PLANTED])
def test_finding_comes_from_its_own_check(tmp_path, toks, monkeypatch, key, tok, trainer, opts, bad, good):
    """Switch the one check off and the test above would fail: the finding is not produced by anything else."""
    class Silent(checks.Tally):
        def hit(self, *a, **k):
            pass

    original = checks.Rows.__init__

    def init(self, *a, **k):
        original(self, *a, **k)
        for name in TALLIES[key]:
            self.t[name] = Silent()
    monkeypatch.setattr(checks.Rows, "__init__", init)
    assert key not in run(tmp_path, toks[tok], bad, trainer, **opts)


def test_severity_of_planted_defects(tmp_path, toks):
    assert ids(run(tmp_path, toks["alpaca"], CLEAN_CHAT)) >= {"missing_end"}
    assert ids(run(tmp_path, toks["chatml"], CLEAN_CHAT, max_length=24)) >= {"truncated"}
    assert run(tmp_path, toks["chatml"], CLEAN_CHAT)["prompt"].level == "warning"


def test_truncation_counts_lost_ends_and_dropped_tokens(tmp_path, toks):
    f = run(tmp_path, toks["chatml"], CLEAN_CHAT, max_length=24)["truncated"]
    assert f.count == len(CLEAN_CHAT)
    assert "lose their end" in f.headline and "of tokens are cut" in f.headline


def test_leakage_between_train_and_validation(tmp_path, toks):
    val = tmp_path / "val.jsonl"
    val.write_text("\n".join(json.dumps(r) for r in [CLEAN_CHAT[0], CLEAN_CHAT[1], chat(900), chat(901)]))
    found = run(tmp_path, toks["chatml"], CLEAN_CHAT, val=str(val))
    assert found["leakage"].count == 2 and found["leakage"].of == 4
    other = tmp_path / "val2.jsonl"
    other.write_text("\n".join(json.dumps(chat(i + 500, answer=f"different text altogether {i} {i * 3}")) for i in range(4)))
    assert "leakage" not in run(tmp_path, toks["chatml"], CLEAN_CHAT, val=str(other))


def test_near_duplicates_are_found_but_distinct_rows_are_not(tmp_path, toks):
    base = " ".join(f"w{i}" for i in range(150))
    rows = [{"text": base}, {"text": base.replace("w70", "changed")}, {"text": " ".join(f"z{i}" for i in range(150))}]
    found = run(tmp_path, toks["chatml"], rows)
    assert found["duplicates"].count == 1 and "nearly" in found["duplicates"].headline


def test_unreadable_and_mixed_rows(tmp_path, toks):
    path = tmp_path / "d.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "hi"}]}\nnot json\n{"text": "x"}\n'
                    + json.dumps(chat(1)) + "\n")
    found = {f.id: f for f in lint(str(path), toks["chatml"]).findings}
    assert found["unreadable"].count == 2 and "mixed" in found


def test_pad_equal_to_end_token_is_a_note_not_an_error(tmp_path, toks):
    found = run(tmp_path, toks["padeos"], CLEAN_CHAT)
    assert found["pad_eos"].level == "note"
    assert "pad_eos" not in run(tmp_path, toks["chatml"], CLEAN_CHAT)


def test_data_shapes(tmp_path, toks):
    rows = [{"conversations": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "hi there"}]},
            {"conversations": [{"from": "system", "value": "be nice"}, {"from": "human", "value": "yo"},
                               {"from": "gpt", "value": "hey"}]}]
    assert "unreadable" not in run(tmp_path, toks["chatml"], rows)
    alpaca = [{"instruction": f"do thing {i}", "input": "with this", "output": f"done {i}"} for i in range(10)]
    assert "unreadable" not in run(tmp_path, toks["chatml"], alpaca)
    parts = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]},
                           {"role": "assistant", "content": "hey"}]}]
    assert "unreadable" not in run(tmp_path, toks["chatml"], parts)


@pytest.mark.parametrize("trainer", ["trl", "mlx", "unsloth", "axolotl", "llama-factory"])
@pytest.mark.parametrize("flavor", ["chatml", "llama"])
def test_clean_data_has_no_errors(tmp_path, toks, trainer, flavor):
    shapes = [CLEAN_CHAT, [{"prompt": f"question {i} about tides", "completion": f"answer {i} about the moon"}
                           for i in range(40)]]
    for rows in shapes:
        found = run(tmp_path, toks[flavor], rows, trainer)
        assert not ids(found), {k: f.headline for k, f in found.items() if f.level == "error"}


def test_text_rows_with_the_end_token_are_clean_in_trl_and_mlx(tmp_path, toks):
    for trainer in ("trl", "mlx"):
        rows = [{"text": f"plain words number {i}"} for i in range(30)]
        assert not ids(run(tmp_path, toks["llama"], rows, trainer))


def test_multi_turn_chat_is_clean(tmp_path, toks):
    rows = [{"messages": [{"role": "user", "content": f"first {i}"}, {"role": "assistant", "content": f"one {i}"},
                          {"role": "user", "content": f"second {i}"}, {"role": "assistant", "content": f"two {i}"}]}
            for i in range(20)]
    for trainer in ("trl", "mlx", "axolotl", "llama-factory", "unsloth"):
        assert not ids(run(tmp_path, toks["chatml"], rows, trainer))


def test_assistant_only_loss_in_trl_needs_generation_markers(tmp_path, toks):
    from neyra.tune.trainers import Unsupported

    with pytest.raises(Unsupported):
        run(tmp_path, toks["chatml"], CLEAN_CHAT, "trl", mask_prompt=True)
    found = run(tmp_path, toks["genmark_ok"], CLEAN_CHAT, "trl", mask_prompt=True)
    assert "prompt" not in found and not ids(found)


def test_gguf_template_mismatch(tmp_path, toks):
    gguf = pytest.importorskip("gguf")
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(toks["chatml"])

    def write(name, template):
        w = gguf.GGUFWriter(str(tmp_path / name), "llama")
        w.add_string("tokenizer.chat_template", template)
        vocab = tk.get_vocab()
        w.add_token_list([t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])])
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return str(tmp_path / name)

    same = write("same.gguf", tinytok.CHATML)
    other = write("other.gguf", tinytok.CHATML.replace("<|im_start|>assistant", "<|im_start|>bot"))
    assert "template_gguf" not in run(tmp_path, toks["chatml"], CLEAN_CHAT, gguf=same)
    found = run(tmp_path, toks["chatml"], CLEAN_CHAT, gguf=other)
    assert found["template_gguf"].level == "error" and "bot" in found["template_gguf"].detail


def test_cli_exit_codes_and_json(tmp_path, toks, capsys):
    bad, good = tmp_path / "bad.jsonl", tmp_path / "good.jsonl"
    bad.write_text("\n".join(json.dumps(r) for r in CLEAN_CHAT))
    good.write_text("\n".join(json.dumps(r) for r in CLEAN_CHAT))
    assert cli.main(["tune", "lint", str(good), toks["chatml"], "--trainer", "mlx", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert cli.main(["tune", "lint", str(bad), toks["alpaca"], "--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["findings"][0]["id"] == "missing_end" and doc["findings"][0]["example"]["tokens"]
    assert cli.main(["tune", "lint", str(bad), toks["chatml"], "--show", "1"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("Nothing blocks training") and "[[" in out


def test_first_line_is_one_sentence(tmp_path, toks, capsys):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in CLEAN_CHAT))
    cli.main(["tune", "lint", str(path), toks["alpaca"]])
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("Do not train yet: ") and first.endswith(".") and first.count(". ") == 0


def test_module_is_importable_without_running():
    assert lint_mod.TRAINERS["trl"].exact and lint_mod.TRAINERS["mlx"].exact
