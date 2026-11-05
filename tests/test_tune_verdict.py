"""The verdict must say each of its sentences for data built to deserve it, and must not say them for data that doesn't."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neyra import cli
from neyra.tune import cases as cases_mod
from neyra.tune import damage, prompts, verdict, verdict_report
from neyra.tune.cases import Case
from neyra.tune.engines import Backends, EngineError, quant_name
from neyra.tune.prompts import Format

N = 100


def make_cases(n=N):
    return [Case(str(i), f"question {i}", f"answer {i}") for i in range(n)]


def hit(i: int, rate: float, salt: int) -> bool:
    """Whether item i is answered right at this rate. Different salts give independent items."""
    return ((i * 2654435761 + salt * 40503) % 1000) / 1000 < rate


def nested(i: int, rate: float) -> bool:
    return (i * 7919 % 1000) / 1000 < rate


def answerer(rate: float, salt: int = 0, is_nested: bool = True, few_rate: float | None = None, general: float = 1.0):
    """A generator that answers question i right at `rate`, or at `few_rate` when the prompt carries examples."""
    def gen(ps):
        out = []
        for p in ps:
            msgs = p.messages
            text = msgs[-1]["content"]
            if text.startswith("question "):
                i = int(text.split()[1])
                r = few_rate if (few_rate is not None and len(msgs) > 2) else rate
                ok = nested(i, r) if is_nested else hit(i, r, salt)
                out.append(f"answer {i}" if ok else "no idea")
            else:
                j = [q for q, _ in damage.QUESTIONS].index(text)
                out.append(damage.QUESTIONS[j][1][0] if nested(j, general) else "no idea")
        return out
    return gen


def backends(base=0.4, few=0.5, tuned=0.9, quants=None, tuned_general=1.0, kl=None, independent=False):
    """Arms whose right answers are nested by default, so every difference runs one way. independent=True mixes them."""
    gens = {"base": answerer(base, 1, not independent, few), "tuned": answerer(tuned, 2, not independent, None, tuned_general)}
    for q, rate in (quants or {}).items():
        gens[q] = answerer(rate)
    return Backends(gens, kl)


def go(b, n=N, margin=0.05, shots=4):
    cs = make_cases(n)
    fmt = Format(True)
    return verdict.run(cs, b, fmt, cases_mod.scorer("exact"), "exact", cs[:shots] and [Case("x", "e", "e")] * shots,
                       margin, "cases.jsonl")


def test_worked():
    r = go(backends(0.4, 0.5, 0.9))
    assert r.verdict == "worked" and r.passed
    assert r.sentence.startswith("This worked")
    assert r.arms["tuned"] > r.arms["few-shot"] > r.arms["base"]


def test_prompt_does_the_same():
    r = go(backends(0.4, 0.89, 0.9), n=200)
    assert r.verdict == "prompt_enough" and not r.passed
    assert "you did not need to train" in r.sentence


def test_prompt_beats_the_tune():
    r = go(backends(0.4, 0.9, 0.6))
    assert r.verdict == "prompt_better" and not r.passed


def test_did_not_work():
    r = go(backends(0.5, 0.5, 0.5))
    assert r.verdict == "did_not_work" and not r.passed
    assert r.sentence.startswith("It did not work")


def test_worse_than_base():
    r = go(backends(0.7, 0.7, 0.3))
    assert r.verdict == "worse_than_base" and not r.passed


def test_gain_lost_in_one_quantization_and_kept_in_another():
    r = go(backends(0.4, 0.5, 0.9, {"Q8_0": 0.9, "Q4_K_M": 0.45}))
    assert r.verdict == "lost_in_quantization" and not r.passed
    assert r.sentence.startswith("The gain is lost at Q4_K_M")
    assert "keeps the gain at Q8_0" in r.sentence


def test_a_file_that_cannot_run_is_reported_and_the_rest_goes_on():
    def broken(ps):
        raise EngineError("llama-server exited with 1 on a.gguf")

    b = backends(0.4, 0.5, 0.9, {"Q8_0": 0.9})
    b.generate["Q4_K_M"] = broken
    r = go(b)
    assert r.verdict == "worked" and r.skipped == [("Q4_K_M", "llama-server exited with 1 on a.gguf")]
    assert "Q4_K_M" not in r.arms


def test_gain_kept_in_every_file():
    r = go(backends(0.4, 0.5, 0.9, {"Q8_0": 0.9, "Q4_K_M": 0.88}))
    assert r.verdict == "worked" and r.passed and "Q4_K_M" in r.sentence


def test_too_few_cases_says_so_instead_of_guessing():
    r = go(backends(0.4, 0.4, 0.6, independent=True), n=30)
    assert r.verdict == "cannot_tell" and not r.passed
    assert "Cannot tell from 30 cases" in r.sentence and "cases would narrow" in r.sentence


def test_fewer_than_ten_cases_is_refused():
    with pytest.raises(cases_mod.CaseError):
        go(backends(), n=8)


def test_damage_to_general_answers_is_reported_beside_a_real_gain():
    r = go(backends(0.4, 0.5, 0.9, tuned_general=0.2))
    assert r.verdict == "worked_with_damage" and r.passed
    assert r.general["kind"] == "worse" and r.general["got_worse"]


def test_cases_that_got_worse_are_listed():
    r = go(backends(0.5, 0.5, 0.9))
    bad = go(backends(0.9, 0.9, 0.5))
    assert r.worse_total == 0
    assert bad.worse_total > 0 and bad.worse[0]["tuned"] == "no idea"


def test_comparison_direction_matters():
    up = verdict.compare("a", "b", np.ones(50), np.zeros(50), 0.05)
    down = verdict.compare("a", "b", np.zeros(50), np.ones(50), 0.05)
    assert (up.kind, down.kind) == ("better", "worse")


def test_equal_requires_the_interval_to_fit_the_margin():
    a = np.array([1.0] * 20 + [0.0] * 10)
    b = np.array([1.0] * 19 + [0.0] * 11)
    assert verdict.compare("a", "b", a, b, 0.05).kind == "unclear"
    assert verdict.compare("a", "b", a, b, 0.3).kind == "same"


def test_scorers():
    c = Case("1", "q", "Paris")
    assert cases_mod.scorer("exact")(" paris ", c) == 1 and cases_mod.scorer("exact")("Paris, France", c) == 0
    assert cases_mod.scorer("contains")("It is Paris, France", c) == 1
    assert cases_mod.scorer("regex:^city: {expected}$")("city: Paris", c) == 1
    assert cases_mod.scorer("regex:^city: {expected}$")("city: Paris.", c) == 0


def test_python_scorer(tmp_path):
    f = tmp_path / "s.py"
    f.write_text("def score(output, expected, case):\n    return 0.5 if expected in output else 0\n")
    assert cases_mod.scorer(f"python:{f}")("Paris", Case("1", "q", "Paris")) == 0.5
    f.write_text("def score(output, expected, case):\n    return 3\n")
    with pytest.raises(cases_mod.CaseError):
        cases_mod.scorer(f"python:{f}")("x", Case("1", "q", "x"))


def test_bad_scorer_and_bad_cases(tmp_path):
    for spec in ("fuzzy", "regex", "regex:(", "python:/nope.py"):
        with pytest.raises(cases_mod.CaseError):
            cases_mod.scorer(spec)
    f = tmp_path / "c.jsonl"
    f.write_text('{"input": "a"}\n')
    with pytest.raises(cases_mod.CaseError, match="expected output"):
        cases_mod.read(str(f))
    f.write_text('{"id": 1, "input": "a", "output": "b"}\n{"id": 1, "input": "c", "output": "d"}\n')
    with pytest.raises(cases_mod.CaseError, match="not unique"):
        cases_mod.read(str(f))


def test_case_shapes(tmp_path):
    f = tmp_path / "c.jsonl"
    rows = [{"prompt": "p", "completion": "c"},
            {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                          {"role": "assistant", "content": "a"}]},
            {"input": "i", "expected": "e"}]
    f.write_text("\n".join(map(json.dumps, rows)))
    got = cases_mod.read(str(f))
    assert [(c.input, c.output, c.system) for c in got] == [("p", "c", ""), ("u", "a", "s"), ("i", "e", "")]


def test_examples_cover_each_label():
    pool = [Case(str(i), f"x{i}", "abc"[i % 3]) for i in range(60)]
    got = cases_mod.examples(pool, 6, seed=1)
    assert sorted(c.output for c in got) == ["a", "a", "b", "b", "c", "c"]


def test_plain_and_chat_prompts():
    shot, case = Case("1", "2+2", "4"), Case("2", "3+3", "6")
    plain = prompts.render(Format(False, "Q: {input}\nA:"), case, [shot])
    assert plain.text == "Q: 2+2\nA: 4\n\nQ: 3+3\nA:"
    chat = prompts.render(Format(True, system="Be brief."), case, [shot])
    assert [m["role"] for m in chat.messages] == ["system", "user", "assistant", "user"]
    assert prompts.cut("4\nmore", ("\n",)) == "4"


def test_quant_names():
    assert quant_name("/m/qwen-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_name("/m/qwen-f16.gguf") == "F16"
    assert quant_name("/m/mine.gguf") == "mine"


def test_text_and_json_output():
    r = go(backends(0.4, 0.5, 0.9, {"Q4_K_M": 0.45}, kl=lambda texts: [0.01] * len(texts)))
    r.skipped.append(("Q2_K", "llama-server is not installed"))
    text = verdict_report.render(r)
    assert text.splitlines()[0] == r.sentence
    assert "not run: llama-server is not installed" in text and "Limits" in text and "KL on 8 neutral texts" in text
    doc = verdict_report.document(r)
    assert doc["verdict"] == "lost_in_quantization" and doc["cases"]["n"] == N
    assert len(doc["per_case_scores"]["tuned"]) == N and doc["not_run"][0]["file"] == "Q2_K"
    json.dumps(doc)


@pytest.fixture
def files(tmp_path):
    cs = make_cases()
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps({"id": c.id, "input": c.input, "output": c.output})
                                                    for c in cs))
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps({"input": f"train {i}", "output": f"answer {i}"})
                                                    for i in range(20)))
    return tmp_path


def test_cli_exit_codes_and_json(files, monkeypatch, capsys):
    from neyra.tune import engines
    state = {"b": backends(0.4, 0.5, 0.9)}
    monkeypatch.setattr(engines, "open_backends", lambda *a, **k: state["b"])
    args = ["tune", "verdict", "base", "tuned", str(files / "cases.jsonl"), "--examples", str(files / "train.jsonl"),
            "--format", "chat", "--json"]
    assert cli.main(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["verdict"] == "worked" and doc["shots"] == 8
    state["b"] = backends(0.5, 0.5, 0.5)
    assert cli.main(args) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "did_not_work"


def test_cli_rejects_unreadable_cases(files, capsys):
    (files / "cases.jsonl").write_text("not json\n")
    code = cli.main(["tune", "verdict", "base", "tuned", str(files / "cases.jsonl"), "--examples",
                     str(files / "train.jsonl")])
    assert code == cli.BAD_INPUT and "not JSON" in capsys.readouterr().err
