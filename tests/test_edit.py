import json

import numpy as np
import pytest

from neyra.cli import main
from neyra.edit import Request, Settings, Words
from neyra.numerics import to_float
from neyra.patch import Patch
from neyra.protocol import bits_digest
from neyra.store import Checkpoint
from neyra.units import UnitSet
from neyra.workflows import edit_facts

tinylm = pytest.importorskip("tinylm", reason="needs MLX", exc_type=ImportError)
TEMPLATE = tinylm.RELATIONS[0][0]


def test_lookup_ends_on_the_subject_token():
    words = Words(tinylm.Tokenizer())
    r = Request("Aldo", "Capital: {} is", "Rome")
    ids, pos = words.lookup(r)
    assert ids[pos] == words.ids("Aldo")[0] and len(ids) == pos + 2


@pytest.fixture(scope="module", params=["dense", "hybrid"])
def world(request, tmp_path_factory):
    model, config, known, _ = tinylm.trained(request.param)
    folder = tmp_path_factory.mktemp(request.param)
    tinylm.save(model, config, folder)
    return request.param, tinylm.runtime(model), Checkpoint.open(str(folder)), known, folder


def requests_for(known, count=2, neighbours=True, rt=None):
    out = []
    words = Words(tinylm.Tokenizer())
    rows = [(words.ids(TEMPLATE.replace("{}", s)), words.target(TEMPLATE, known[(0, s)])) for s in tinylm.SUBJECTS]
    right = [s for s, ok in zip(tinylm.SUBJECTS, rt.score(rows)[1]) if ok] if rt else tinylm.SUBJECTS
    for s in right[:count]:
        old = known[(0, s)]
        new = next(o for o in tinylm.CAPITALS if o != old)
        near = tuple(TEMPLATE.replace("{}", o) for o in right[count:] if known[(0, o)] == old)
        out.append(Request(s, TEMPLATE, new, old, neighbours=near if neighbours else ()))
    return out


def neutral_file(tmp_path):
    rng = np.random.default_rng(1)
    words = tinylm.SUBJECTS + tinylm.CAPITALS
    path = tmp_path / "neutral.txt"
    path.write_text(" ".join(str(rng.choice(words)) for _ in range(6000)))
    return path


SETTINGS = {"layers": [1, 2, 3], "contexts": 0, "steps": 60, "lr": 1.0, "clamp": 30.0, "anchor": 30.0,
            "lam": 50.0, "neutral": 60, "held": 10, "window": 8}


def test_edit_takes_and_the_patch_is_exact(world, tmp_path):
    _, rt, ck, known, _ = world
    reqs = requests_for(known, rt=rt)
    done = edit_facts(rt, ck, reqs, Settings(**SETTINGS), neutral=str(neutral_file(tmp_path)))
    r = done.report
    assert r["before"]["efficacy"] == 0.0 and r["after"]["efficacy"] == 1.0
    assert r["after"]["exact"] == 1.0
    assert all(e.low_rank and len(e.idx) == 0 for e in done.patch.edits)
    assert r["gates"]["exact_revert"]
    # the patch rebuilds the very bits the engine measured, and the inverse restores the base
    edited = done.patch.apply(ck)
    for e in done.patch.edits:
        assert bits_digest(e.dtype, edited[e.name]) == e.result_digest != e.base_digest
        assert np.array_equal(e.apply(ck.tensors[e.name].read()), edited[e.name])
    inverse = done.patch.invert(ck)
    for e in inverse.edits:
        assert np.array_equal(e.apply(edited[e.name]), ck.tensors[e.name].read())


def test_cli_edit_signs_and_every_command_accepts_the_patch(world, tmp_path, capsys, monkeypatch):
    _, rt, _, known, folder = world
    monkeypatch.setattr("neyra.runtime.open_runtime", lambda *a, **k: rt)
    reqs = requests_for(known, rt=rt)
    path = tmp_path / "requests.json"
    path.write_text(json.dumps([{"subject": r.subject, "prompt": r.prompt, "target": r.target, "true": r.true,
                                 "neighbours": list(r.neighbours)} for r in reqs]))
    key = tmp_path / "k.key"
    assert main(["keygen", "-o", str(key)]) == 0
    out = tmp_path / "edit.neyra"
    args = ["edit", str(folder), str(path), "-o", str(out), "--key", str(key), "--layers", "1,2,3", "--contexts", "0",
            "--steps", "60", "--lr", "1", "--lam", "50", "--text", str(neutral_file(tmp_path)), "--neutral", "60", "--held", "10",
            "--window", "8"]
    code = main(args)
    text = capsys.readouterr().out
    assert "efficacy" in text and "verdict" in text
    assert out.exists() and out.with_suffix(".receipt.json").exists() and out.with_suffix(".ledger.jsonl").exists()
    if code == 0:
        verify = ["verify", str(out.with_suffix(".receipt.json")), "--ledger", str(out.with_suffix(".ledger.jsonl")),
                  "--patch", str(out), "--key", str(key.with_suffix(".pub"))]
        assert main(verify) == 0
        assert "signature verified" in capsys.readouterr().out
    assert main(["apply", str(folder), str(out), "-o", str(tmp_path / "model")]) == 0
    assert main(["export", str(folder), str(out), "--format", "peft", "-o", str(tmp_path / "lora")]) == 0
    other = Patch.load(out)
    assert other.id == Patch.load(out).id


def test_no_neighbours_means_no_claim(world, tmp_path):
    _, rt, ck, known, _ = world
    done = edit_facts(rt, ck, requests_for(known, neighbours=False, rt=rt), Settings(**SETTINGS),
                      neutral=str(neutral_file(tmp_path)))
    assert done.report["gates"]["specificity"] is False
    assert done.verdict == "FAIL" and done.claim["verdict"] == "FAIL"


def test_patch_leaves_untouched_tensors_alone(world, tmp_path):
    _, rt, ck, known, _ = world
    done = edit_facts(rt, ck, requests_for(known, 1, rt=rt), Settings(**SETTINGS), neutral=str(neutral_file(tmp_path)))
    assert {e.name for e in done.patch.edits} == {n for n in ck.tensors if n.endswith("mlp.down_proj.weight")
                                                  and any(f"layers.{i}." in n for i in (1, 2, 3))}
    name = done.patch.edits[0].name
    w0 = to_float(ck.tensors[name].read(), "BF16")
    w1 = to_float(done.patch.apply(ck)[name], "BF16")
    assert np.linalg.norm(w1 - w0) > 0


def test_writing_outside_dense_mlp_is_refused(world, tmp_path):
    _, rt, ck, known, _ = world
    with pytest.raises(ValueError, match="no dense MLP"):
        edit_facts(rt, ck, requests_for(known, rt=rt), Settings(**{**SETTINGS, "layers": [9]}),
                   neutral=str(neutral_file(tmp_path)))


def test_edit_patches_merge_and_split_like_any_other(world, tmp_path, capsys):
    _, rt, ck, known, folder = world
    reqs = requests_for(known, 2, rt=rt)
    paths = []
    for i, r in enumerate(reqs):
        done = edit_facts(rt, ck, [r], Settings(**SETTINGS), neutral=str(neutral_file(tmp_path)))
        done.patch.save(tmp_path / f"{i}.neyra")
        paths.append(str(tmp_path / f"{i}.neyra"))
    assert main(["merge", str(folder), *paths, "-o", str(tmp_path / "both.neyra")]) in (0, 1)
    merged = Patch.load(tmp_path / "both.neyra")
    assert merged.entries > 0
    assert len(UnitSet(merged, ck, "column")) > 0
    capsys.readouterr()
    assert main(["show", paths[0]]) == 0


def test_recomputing_layers_gives_the_same_gradient(world):
    from neyra.runtime import DeltaBatch

    _, rt, _, known, _ = world
    words = Words(tinylm.Tokenizer())
    r = requests_for(known, 1, rt=rt)[0]
    ids, pos = words.lookup(r)
    target = words.target(r.text(), r.target)
    seq = ids + target[:-1]
    cols = np.array([len(ids) - 1 + j for j in range(len(target))])
    batch = DeltaBatch(np.array([seq]), np.array([0]), np.array([pos]), np.zeros(len(cols), int), cols,
                       np.array(target), np.full(len(cols), 1 / len(cols)), np.zeros((0, tinylm.VOCAB), np.float32))
    delta = np.random.default_rng(0).standard_normal((1, 96)) * 50
    a, ga = rt.delta_loss(1, delta, batch, recompute=False)
    b, gb = rt.delta_loss(1, delta, batch, recompute=True)
    assert np.allclose(a, b, rtol=1e-4) and np.allclose(ga, gb, rtol=1e-3, atol=1e-6 * np.abs(ga).max())
    assert np.abs(ga).max() > 0


def test_restoring_the_last_layer_at_an_earlier_position_changes_nothing(world):
    from neyra.edit import next_token
    from neyra.runtime import Hooks

    _, rt, _, _, _ = world
    words = Words(tinylm.Tokenizer())
    ids = words.ids("Capital: Aldo is")
    n = len(rt.layers())
    clean = rt.gather([ids], list(range(n)), "out", [1])
    noise = Hooks(noise=(np.array([1]), np.random.default_rng(0).standard_normal((1, 96)) * 3))
    noisy = next_token(rt, ids, noise)
    last = Hooks(noise=noise.noise, put={n - 1: (np.array([1]), clean[n - 1])})
    assert np.allclose(next_token(rt, ids, last), noisy, atol=1e-6)
    again = Hooks(put={n - 2: (np.array([1]), clean[n - 2])})
    assert np.allclose(next_token(rt, ids, again), next_token(rt, ids), atol=1e-5)


def test_a_saved_editor_carries_on_where_it_stopped(world, tmp_path):
    from neyra.edit import Editor, second_moments

    _, rt, ck, known, _ = world
    reqs = requests_for(known, 4, rt=rt)
    words = Words(tinylm.Tokenizer())
    windows = [words.ids("Capital: Aldo Rome . Tongue: Brem Latin")] * 8
    layers = [1, 2, 3]
    cov = second_moments(rt, windows, layers)
    settings = Settings(**{**SETTINGS, "contexts": 0})

    rt.reset()
    whole = Editor(rt, ck, settings, layers, cov, say=lambda *_: None)
    whole.write(reqs[:2])
    whole.write(reqs[2:])
    straight = {n: b for n, (b, _) in whole.stack.tensors().items()}

    rt.reset()
    first = Editor(rt, ck, settings, layers, cov, say=lambda *_: None)
    first.write(reqs[:2])
    first.save(tmp_path / "state.npz")
    rt.reset()
    second = Editor(rt, ck, settings, layers, cov, say=lambda *_: None)
    second.load(tmp_path / "state.npz")
    second.write(reqs[2:])
    for name, bits in second.stack.tensors().items():
        assert np.array_equal(bits[0], straight[name])
    rt.reset()
