import json

import numpy as np
import pytest
from conftest import bf16, symlink
from fakes import NAME, FakeRuntime

from neyra.lab import BudgetSpent, Item, Lab, Refused, Task, read_jsonl
from neyra.ledger import verify
from neyra.numerics import to_float
from neyra.patch import Patch, diff_tensor
from neyra.repair import project_columns

SECRET_MARKER = "holdout-secret-question"


def setup(tmp_path, model_dir, rng, passes=10, submissions=1, alternatives=(), **lab_args):
    w = bf16(rng.standard_normal((32, 48)) * 0.1)
    d = model_dir("m", {NAME: w, "model.embed_tokens.weight": bf16(rng.standard_normal((16, 32)))})
    u = rng.standard_normal(32)
    u /= np.linalg.norm(u)
    dev = [Item(f"d{i}", f"dev {i}", "4") for i in range(24)]
    held = [Item(f"h{i}", f"{SECRET_MARKER} {i}", "4") for i in range(64)]
    task = Task("t", dev, held, alternatives=list(alternatives), max_tokens=64)
    return Lab(tmp_path / "lab", str(d), task, FakeRuntime(w, u), passes, submissions, **lab_args), w, u


def real_edit(lab, w, u, tmp_path):
    bits = project_columns(w, "BF16", u, np.arange(48), 1.0).reshape(w.shape)
    p = lab.root / "good.neyra"
    Patch([diff_tensor(NAME, "BF16", w, bits)]).bind(lab.table).save(p)
    return lab.propose_from_file(str(p))["candidate"]


def test_good_edit_passes_and_receipt_chains(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    c = real_edit(lab, w, u, tmp_path)
    r = lab.submit(c)
    assert r["verdict"] == "PASS", r["gates"]
    head, _ = verify(lab.root / "ledger.jsonl")
    assert r["ledger_head"] == head
    assert json.loads((lab.root / f"{c}.claim.json").read_text())["patch_id"] == r["patch_id"]


def test_irrelevant_edit_fails_the_verdict(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    v = rng.standard_normal(32)
    v -= (v @ u) * u
    v /= np.linalg.norm(v)
    bits = project_columns(w, "BF16", v, np.arange(48), 1.0).reshape(w.shape)
    p = lab.root / "bad.neyra"
    Patch([diff_tensor(NAME, "BF16", w, bits)]).bind(lab.table).save(p)
    r = lab.submit(lab.propose_from_file(str(p))["candidate"])
    assert r["verdict"] == "FAIL" and not r["gates"]["effect"]


def test_holdout_text_never_leaves_the_lab(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    c = real_edit(lab, w, u, tmp_path)
    out = json.dumps([lab.status(), lab.sample(), lab.compare(c), lab.control(c), lab.submit(c)], default=str)
    assert SECRET_MARKER not in out
    assert SECRET_MARKER not in (lab.root / "ledger.jsonl").read_text()


def test_budget_is_enforced(tmp_path, model_dir, rng):
    lab, *_ = setup(tmp_path, model_dir, rng, passes=2)
    lab.sample()
    lab.sample()
    with pytest.raises(BudgetSpent):
        lab.sample()


def test_only_one_submission(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    c = real_edit(lab, w, u, tmp_path)
    lab.submit(c)
    with pytest.raises(BudgetSpent):
        lab.submit(c)


def test_refuses_protected_tensors_and_scaling(tmp_path, model_dir, rng):
    lab, w, _u = setup(tmp_path, model_dir, rng)
    emb = lab.ck.tensors["model.embed_tokens.weight"].read()
    changed = emb.copy()
    changed[0, 0] += 3
    p = lab.root / "emb.neyra"
    Patch([diff_tensor("model.embed_tokens.weight", "BF16", emb, changed)]).bind(lab.table).save(p)
    with pytest.raises(Refused):
        lab.propose_from_file(str(p))
    scaled = bf16(to_float(w, "BF16") * 0.5)
    p2 = lab.root / "s.neyra"
    Patch([diff_tensor(NAME, "BF16", w, scaled)]).bind(lab.table).save(p2)
    with pytest.raises(Refused):
        lab.propose_from_file(str(p2))


def test_free_alternative_can_fail_the_verdict(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng, alternatives=[{"name": "same as edit", "system": "x"}])
    c = real_edit(lab, w, u, tmp_path)
    original = lab._run

    def rigged(items, seed, **kw):
        if "system" in kw:
            lab._apply(c)
            scores = original(items, seed, **kw)
            lab.rt.reset()
            return scores
        return original(items, seed, **kw)

    lab._run = rigged
    r = lab.submit(c)
    assert not r["gates"]["beats_free_alternatives"] and r["verdict"] == "FAIL"


def test_control_keeps_size_of_change(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    c = real_edit(lab, w, u, tmp_path)
    ctrl = lab.control_patch(c).edits[0]
    real = lab.candidate(c).patch.edits[0]

    def norm(e):
        f = to_float(e.apply(w, verify=False), "BF16").astype(np.float64)
        return np.linalg.norm(f - to_float(w, "BF16"))

    assert norm(ctrl) == pytest.approx(norm(real), rel=0.03)


def test_several_submissions_reveal_one_bit_and_stop_at_pass(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng, submissions=3)
    v = rng.standard_normal(32)
    v -= (v @ u) * u
    v /= np.linalg.norm(v)
    bad = lab.root / "bad.neyra"
    Patch([diff_tensor(NAME, "BF16", w, project_columns(w, "BF16", v, np.arange(48), 1.0).reshape(w.shape))]).bind(lab.table).save(bad)
    r = lab.submit(lab.propose_from_file(str(bad))["candidate"])
    assert r == {"verdict": "FAIL", "submissions_left": 2}
    good = real_edit(lab, w, u, tmp_path)
    assert lab.submit(good)["verdict"] == "PASS"
    with pytest.raises(BudgetSpent):
        lab.submit(good)


def test_agent_cannot_reach_outside_the_lab(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    secret = tmp_path / "private.neyra"
    Patch([diff_tensor(NAME, "BF16", w, project_columns(w, "BF16", u, np.arange(48), 1.0).reshape(w.shape))]).bind(lab.table).save(secret)
    for path in (str(secret), str(lab.root / ".." / "private.neyra"), "/etc/passwd"):
        with pytest.raises(Refused):
            lab.propose_from_file(path)
    link = lab.root / "link.neyra"
    symlink(link, secret)
    with pytest.raises(Refused):
        lab.propose_from_file(str(link))


def test_allowed_directory_is_opt_in(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    extra = tmp_path / "work"
    extra.mkdir()
    lab.roots.append(extra.resolve())
    p = extra / "e.neyra"
    Patch([diff_tensor(NAME, "BF16", w, project_columns(w, "BF16", u, np.arange(48), 1.0).reshape(w.shape))]).bind(lab.table).save(p)
    assert lab.propose_from_file(str(p))["candidate"] == "c0"


def test_jsonl_survives_unicode_line_separators(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"problem": "a\u2028b"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_jsonl(p) == [{"problem": "a\u2028b"}]


def claim(**kw):
    return {"family": "CANARY", "tensors": [NAME], "effect_metric": "defect_reduction", "effect_estimate": 0.5,
            "confidence": 0.7, "status": "partial", **kw}


def test_chat_costs_a_pass_and_caps_samples(tmp_path, model_dir, rng):
    lab, *_ = setup(tmp_path, model_dir, rng, passes=2)
    out = lab.chat("hello there", n=3)
    assert len(out["replies"]) == 3 and lab.status()["passes_left"] == 1
    with pytest.raises(Refused):
        lab.chat("hello", n=99)
    lab.chat("again")
    with pytest.raises(BudgetSpent):
        lab.chat("once more")


def test_report_is_final_and_lands_in_the_ledger(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    c = real_edit(lab, w, u, tmp_path)
    r = lab.report([claim()])
    head, _ = verify(lab.root / "ledger.jsonl")
    assert r["ledger_head"] == head
    assert '"status":"partial"' in (lab.root / "ledger.jsonl").read_text()
    for call in (lambda: lab.report([claim()]), lambda: lab.submit(c), lambda: lab.sample()):
        with pytest.raises(BudgetSpent):
            call()


@pytest.mark.parametrize("bad", [{"status": "great"}, {"confidence": 1.5}, {"confidence": True}, {"tensors": NAME},
                                 {"effect_estimate": "high"}, {"family": ""}])
def test_malformed_claims_are_refused_and_can_be_retried(tmp_path, model_dir, rng, bad):
    lab, *_ = setup(tmp_path, model_dir, rng)
    with pytest.raises(Refused):
        lab.report([claim(**bad)])
    with pytest.raises(Refused):
        lab.report([{k: v for k, v in claim().items() if k != "status"}])
    with pytest.raises(Refused):
        lab.report([])
    assert lab.report([claim(effect_estimate=None)])["claims"] == 1


@pytest.mark.parametrize("withheld", [False, True])
def test_withheld_mode_is_visible_and_blocks_diffing(tmp_path, model_dir, rng, withheld):
    lab, *_ = setup(tmp_path, model_dir, rng, withheld=withheld)
    assert lab.status()["mode"] == ("W" if withheld else "B")
    if withheld:
        with pytest.raises(Refused):
            lab.require_base()
    else:
        lab.require_base()


class Measuring(FakeRuntime):
    """Scores a text by how many weight columns differ from the base."""

    def logprobs(self, tokens, start=1, positions=None):
        out = np.zeros(len(tokens) - start if positions is None else len(positions))
        if NAME in self.current:
            bits, dtype = self.current[NAME]
            out[: len(out) // 2] = -np.count_nonzero((to_float(bits, dtype) != self.base).any(axis=0))
        return out


def test_bisect_says_what_it_measured(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    lab.rt = Measuring(w, u)
    c = real_edit(lab, w, u, tmp_path)
    r = lab.bisect(c, lab.sample()["rollouts"])
    assert r["blame"] == "insertion"
    entry = next(e for e in read_jsonl(lab.root / "ledger.jsonl") if e["kind"] == "bisect")
    assert entry["body"]["blame"] == "insertion"
    assert lab.candidate(r["candidate"]).patch.meta["blame"] == "insertion"


def test_receipt_counts_sessions_in_its_ledger_and_says_what_is_only_asserted(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    first = lab.submit(real_edit(lab, w, u, tmp_path))
    assert first["attempts"] == {"counted": 1, "unit": "lab sessions",
                                 "counted_in": "the ledger this receipt is bound to", "complete": False}
    assert any("clock" in line for line in first["basis"]["asserted"])
    assert any("applying patch_id" in line for line in first["basis"]["exact"])
    again = Lab(lab.root, lab.ck.spec, lab.task, FakeRuntime(w, u), 10, 1)
    assert again.submit(real_edit(again, w, u, tmp_path))["attempts"]["counted"] == 2
